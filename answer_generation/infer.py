import argparse
import os
import sys
import numpy as np
from pathlib import Path

from libs import (
    EvaluationMetrics,
    load_prepared_data,
    create_output_directory,
    save_config_snapshot,
    save_prompt_template,
    save_summary_results,
    run_batch_inference,
    load_config,
    setup_logging,
    create_model,
    InferenceResultsLoader,
)


def main():
    parser = argparse.ArgumentParser(
        description="Video Question Answering Inference with Evaluation"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single", action="store_true", help="Single video inference mode"
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch inference mode on prepared JSON files",
    )

    # Single mode arguments
    parser.add_argument(
        "--video_path", "-v", type=str, help="Path to the video file (single mode)"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask about the video (single mode)",
    )

    # Batch mode arguments
    parser.add_argument(
        "--prepared_folder",
        "-p",
        type=str,
        help="Path to folder containing prepared JSON files (batch mode)",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start index for batch processing (0-based, inclusive). Enables parallel processing by dividing dataset.",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index for batch processing (0-based, exclusive). Use with --start for parallel processing.",
    )

    # Common arguments
    parser.add_argument(
        "--output", "-o", type=str, help="Output file to save the result"
    )
    parser.add_argument(
        "--skip_processed",
        action="store_true",
        help="Skip files that already have a result in the output directory (batch mode).",
    )

    args = parser.parse_args()

    # Set up enhanced logging with custom configuration
    output_dir_path = None
    if args.batch and args.output:
        # For batch mode, create log directory within output directory
        output_dir_path = Path(args.output).parent if args.output != "." else Path.cwd()
        log_dir = output_dir_path / "logs"
    else:
        log_dir = None

    logger = setup_logging(
        name="VQA_Inference",
        log_dir=str(log_dir) if log_dir else None,
        console_level="INFO",
        file_level="DEBUG",
    )

    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Log configuration details with enhanced formatting
    logger.success("Configuration loaded successfully!")
    logger.info(f"Model type: {config.get('model_type', 'transformers')}")
    logger.info(f"Model name: {config.get('model_name', 'Unknown')}")

    # Add context logger for this session
    session_logger = logger.with_context(
        config_file=args.config,
        model_type=config.get("model_type", "transformers"),
        model_name=config.get("model_name", "Unknown"),
    )

    # Validate arguments based on mode
    if args.single:
        if not args.video_path:
            logger.error("--video_path is required for single mode")
            sys.exit(1)
        if not os.path.exists(args.video_path):
            logger.error(f"Video file not found: {args.video_path}")
            sys.exit(1)

    elif args.batch:
        if not args.prepared_folder:
            logger.error("--prepared_folder is required for batch mode")
            sys.exit(1)
        if not os.path.exists(args.prepared_folder):
            logger.error(f"Prepared folder not found: {args.prepared_folder}")
            sys.exit(1)
        
        # Validate start/end arguments for parallel processing
        if args.start is not None and args.start < 0:
            logger.error("--start must be >= 0")
            sys.exit(1)
        if args.end is not None and args.end <= 0:
            logger.error("--end must be > 0")
            sys.exit(1)
        if args.start is not None and args.end is not None and args.start >= args.end:
            logger.error("--start must be less than --end")
            sys.exit(1)

    try:
        # Create model with timing
        logger.info("Initializing model...")
        with logger.time_operation("model_initialization"):
            model = create_model(config)

        logger.success("Model initialized successfully!")

        if args.single:
            # Single video inference (original functionality)
            question = (
                args.question
                if args.question
                else config.get("prompt", "Describe this video.")
            )

            logger.info(f"Running inference on video: {args.video_path}")
            logger.info(f"Question: {question}")

            trace_id = f"inference_{os.path.basename(args.video_path)}_{hash(question) % 10000}"
            result = model.infer(args.video_path, question, logger, trace_id)

            # Output result
            print(f"\nResult: {result}")

            # Save to file if specified
            if args.output:
                with open(args.output, "w") as f:
                    f.write(f"Video: {args.video_path}\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Answer: {result}\n")
                logger.info(f"Result saved to: {args.output}")

        elif args.batch:
            # Batch inference with evaluation
            logger.info("Starting batch inference mode...")

            with logger.time_operation("batch_inference_setup"):
                logger.info("Initializing evaluation metrics...")
                evaluator = EvaluationMetrics()

                logger.info(f"Loading prepared data from: {args.prepared_folder}")
                prepared_data = load_prepared_data(Path(args.prepared_folder))
                logger.success(f"Loaded {len(prepared_data)} prepared samples")

                if not prepared_data:
                    logger.error("No prepared data found")
                    sys.exit(1)

                # Apply start/end slicing for parallel processing
                original_count = len(prepared_data)
                if args.start is not None or args.end is not None:
                    start_idx = args.start if args.start is not None else 0
                    end_idx = args.end if args.end is not None else len(prepared_data)
                    
                    # Validate indices against actual data size
                    if start_idx >= len(prepared_data):
                        logger.error(f"--start ({start_idx}) is >= dataset size ({len(prepared_data)})")
                        sys.exit(1)
                    if end_idx > len(prepared_data):
                        logger.warning(f"--end ({end_idx}) exceeds dataset size ({len(prepared_data)}), clamping to dataset size")
                        end_idx = len(prepared_data)
                    
                    prepared_data = prepared_data[start_idx:end_idx]
                    logger.info(f"Processing slice [{start_idx}:{end_idx}] = {len(prepared_data)} samples out of {original_count} total")

                # Create output directory with slice info if using parallel processing
                if args.start is not None or args.end is not None:
                    start_idx = args.start if args.start is not None else 0
                    end_idx = args.end if args.end is not None else original_count
                    slice_suffix = f"_slice_{start_idx}_{end_idx}"
                    if args.output.endswith('/'):
                        output_path = args.output.rstrip('/') + slice_suffix
                    else:
                        output_path = args.output + slice_suffix
                else:
                    output_path = args.output
                
                output_dir = create_output_directory(output_path)
                logger.info(f"Results will be saved to: {output_dir}")

                # Save configuration snapshot first
                logger.info("Saving configuration snapshot...")
                save_config_snapshot(config, args.config, output_dir)

                # Save prompt template used for this inference
                logger.info("Saving prompt template...")
                save_prompt_template(config, output_dir, model)

            # Run batch inference with timing
            logger.info("Starting batch inference processing...")
            with logger.time_operation("batch_inference_processing"):
                new_results = run_batch_inference(
                    model,
                    prepared_data,
                    logger,
                    evaluator,
                    output_dir,
                    config,
                    args.config,
                    args,
                )

            # Load existing results if skip_processed is used
            all_results = []
            if args.skip_processed:
                logger.info("Loading existing individual results...")
                try:
                    results_loader = InferenceResultsLoader(output_dir)
                    existing_results = results_loader.load_individual_results()
                    logger.info(f"Loaded {len(existing_results)} existing results")
                    all_results.extend(existing_results)
                except Exception as e:
                    logger.warning(f"Could not load existing results: {str(e)}")

            # Add new results
            all_results.extend(new_results)

            # Remove duplicates based on file_stem (prioritize new results over existing ones)
            unique_results = {}
            # Add existing results first
            for result in all_results:
                file_stem = result.get("file_stem")
                if file_stem and file_stem not in unique_results:
                    unique_results[file_stem] = result

            # Override with new results if they exist (ensuring new results take precedence)
            for result in new_results:
                file_stem = result.get("file_stem")
                if file_stem:
                    unique_results[file_stem] = result

            results = list(unique_results.values())
            logger.info(f"Total unique results after combining: {len(results)}")

            if not results:
                logger.warning(
                    "No results found (neither new nor existing). Summary might be empty."
                )

            # Update command line args in all results (both existing and new)
            for result in results:
                result["command_line_args"] = vars(args)

            # Save summary results
            logger.info("Generating summary statistics...")
            save_summary_results(results, config, output_dir, args)

            # Print final summary
            logger.success("Batch inference completed successfully!")
            logger.info(f"Processed {len(new_results)} new videos in this run.")
            logger.info(f"Total results (including existing): {len(results)}")
            logger.info(f"Results saved to: {output_dir}")

            if results and results[0].get("metrics"):
                avg_sts = np.mean([r["metrics"]["sts_score"] for r in results])
                avg_meteor = np.mean([r["metrics"]["meteor_score"] for r in results])
                avg_bert_f1 = np.mean([r["metrics"]["bert_f1"] for r in results])
                avg_ndcg = np.mean([r["metrics"]["ndcg_score"] for r in results])

                logger.success("Average Performance (all results):")
                logger.info(f"  📊 STS Score: {avg_sts:.4f}")
                logger.info(f"  🎯 METEOR Score: {avg_meteor:.4f}")
                logger.info(f"  🤖 BERTScore F1: {avg_bert_f1:.4f}")
                logger.info(f"  📈 NDCG Score: {avg_ndcg:.4f}")

    except Exception as e:
        logger.exception(f"Critical error during inference: {str(e)}")
        logger.error("Inference failed. Check logs for detailed traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Example usage:
# Single process:
# python infer.py --config configs/videollama3.yaml --batch --video_path ../data/updated_test/videos/ --prepared_folder ../prepared/f_AG_test_whisper_v2/ --output results/test_final_vl3

# Parallel processing with multiple processes (divide 100 videos into 4 chunks):
# Process 1: python infer.py --config configs/videollama3.yaml --batch --video_path ../data/updated_test/videos/ --prepared_folder ../prepared/f_AG_test_whisper_v2/ --output results/test_final_vl3 --start 0 --end 25
# Process 2: python infer.py --config configs/videollama3.yaml --batch --video_path ../data/updated_test/videos/ --prepared_folder ../prepared/f_AG_test_whisper_v2/ --output results/test_final_vl3 --start 25 --end 50
# Process 3: python infer.py --config configs/videollama3.yaml --batch --video_path ../data/updated_test/videos/ --prepared_folder ../prepared/f_AG_test_whisper_v2/ --output results/test_final_vl3 --start 50 --end 75
# Process 4: python infer.py --config configs/videollama3.yaml --batch --video_path ../data/updated_test/videos/ --prepared_folder ../prepared/f_AG_test_whisper_v2/ --output results/test_final_vl3 --start 75 --end 100