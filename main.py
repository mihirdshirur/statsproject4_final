
import argparse
import sys
import pathlib
import my_utils

def clear_weather_csvs():
    weather_dir = pathlib.Path("data/weather")
    if not weather_dir.exists():
        print("Directory does not exist:", weather_dir)
        return
    for item in weather_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            item.unlink()
            print(f"Deleted cached weather file: {item}")
    print("Finished removing all cached weather data")

def clear_fresh_pjm_csv():
    fresh_file = pathlib.Path("data/pjm/hrl_load_metered_fresh.csv")
    if fresh_file.is_file():
        fresh_file.unlink()
        print(f"Deleted cached fresh pjm file: {fresh_file}")
    print("Finished removing cached fresh pjm file")

def clear_complete_csvs():
    complete_dir = pathlib.Path("data/complete_dfs")
    if not complete_dir.exists():
        print("Directory does not exist:", complete_dir)
        return
    for item in complete_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".csv":
            item.unlink()
            print(f"Deleted cached complete df: {item}")
    print("Finished removing all cached complete dfs")

def clear_models():
    model_dir = pathlib.Path("models")
    if not model_dir.exists():
        print("Directory does not exist:", model_dir)
        return
    for item in model_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".pkl":
            item.unlink()
            print(f"Deleted cached model: {item}")
    print("Finished removing all cached models")


def download_weather_csvs():
    my_utils.makeDirectories(verbose=True)
    my_utils.getAllWeatherDfs(verbose=True)

def download_fresh_pjm_csv():
    my_utils.makeDirectories(verbose=True)
    # just requesting one particular load_area will trigger
    # a download if not cached
    my_utils.getEnergyDf("AECO", verbose=True)

def construct_complete_csvs():
    my_utils.makeDirectories(verbose=True)
    # these are automatically cached like everything else
    my_utils.getAllCompleteDfs(verbose=True)






def run_clean():
    print("Running make clean...")
    clear_weather_csvs()
    clear_fresh_pjm_csv()
    clear_complete_csvs()
    clear_models()
    print("Finished running make clean")

def run_train():
    print("Running make (e.g. training models)...")
    my_utils.trainAll(force_refresh=True)
    print("Finished running make (e.g. training models)")

def run_predictions():
    # Note that this method is sensitive because it must only output
    # the specific sequence described in the spec
    my_utils.predictAll()
    
def run_rawdata():
    print("Running make rawdata...")
    clear_weather_csvs()
    clear_fresh_pjm_csv()
    clear_complete_csvs()

    download_weather_csvs()
    download_fresh_pjm_csv()
    construct_complete_csvs()
    print("Finished running make rawdata")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline entrypoint: clean, predictions, train, or rawdata"
    )
    parser.add_argument(
        "task",
        choices=["clean", "predictions", "train", "rawdata"],
        help="Which step of the pipeline to run"
    )
    args = parser.parse_args()

    if args.task == "clean":
        run_clean()
    elif args.task == "predictions":
        run_predictions()
    elif args.task == "train":
        run_train()
    elif args.task == "rawdata":
        run_rawdata()
    else:
        print("Unrecognized command.", file=sys.stderr)
        sys.exit(1)
    
if __name__ == "__main__":
    main()

