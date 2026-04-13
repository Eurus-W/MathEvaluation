import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--data_names", required=True, type=str)
    parser.add_argument("--prompt_type", required=True, type=str)
    parser.add_argument("--temperature", required=True, type=str)
    parser.add_argument("--n_sampling", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def find_metrics_file(data_dir, seed, temperature):
    pattern = os.path.join(
        data_dir,
        f"*seed{seed}_t{temperature}_*_*metrics.json",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find metrics file for seed={seed}, temperature={temperature} under {data_dir}"
        )
    return matches[0]


def main():
    args = parse_args()
    data_names = [item.strip() for item in args.data_names.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "prompt_type": args.prompt_type,
        "temperature": float(args.temperature),
        "n_sampling": args.n_sampling,
        "seeds": seeds,
        "data_names": {},
        "overall": {},
    }

    overall_runs = {seed: {"acc": [], "avg_output_tokens": []} for seed in seeds}

    for data_name in data_names:
        data_dir = os.path.join(args.output_dir, data_name)
        runs = []
        for seed in seeds:
            metrics_path = find_metrics_file(data_dir, seed, args.temperature)
            metrics = load_json(metrics_path)
            run_result = {
                "seed": seed,
                "metrics_path": metrics_path,
                "acc": metrics["acc"],
                "avg_output_tokens": metrics["avg_output_tokens"],
            }
            runs.append(run_result)
            overall_runs[seed]["acc"].append(metrics["acc"])
            overall_runs[seed]["avg_output_tokens"].append(metrics["avg_output_tokens"])

        summary["data_names"][data_name] = {
            "runs": runs,
            "avg_acc": mean([item["acc"] for item in runs]),
            "avg_output_tokens": mean([item["avg_output_tokens"] for item in runs]),
        }

    overall_run_list = []
    for seed in seeds:
        overall_run_list.append(
            {
                "seed": seed,
                "avg_acc": mean(overall_runs[seed]["acc"]),
                "avg_output_tokens": mean(overall_runs[seed]["avg_output_tokens"]),
            }
        )

    summary["overall"] = {
        "runs": overall_run_list,
        "avg_acc": mean([item["avg_acc"] for item in overall_run_list]),
        "avg_output_tokens": mean(
            [item["avg_output_tokens"] for item in overall_run_list]
        ),
    }

    output_path = os.path.join(
        args.output_dir,
        f"repeat_summary_t{args.temperature}_n{args.n_sampling}_seeds_{args.seeds.replace(',', '-')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(json.dumps(summary, indent=4, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
