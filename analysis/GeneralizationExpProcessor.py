import os
import json
import argparse
import numpy as np
from tools import plot_learning_curve

class GeneralizationExpProcessor:
    """
    Custom class to analyze and visualiza the cumulative experimental results

    # Arguments
        :param to_dir: (string)
        :param path: (string) absolute path to 'experiment.json' file if already exists
    """

    def __init__(self,
                 config,
                 to_dir=".",
                 include_std=True,
                 image_format="pdf",
                 path="generalization.json"):
        self.config = config
        self.to_dir = to_dir
        self.include_std = include_std
        self.image_format = image_format
        self.path = path
        self.hist = {}

        if os.path.isfile(self.path):
            self.load_data()

        if not os.path.exists(self.to_dir):
            os.makedirs(self.to_dir)

    def save_data(self):
        """
        Overwrites the experiment history
        """
        with open(self.path, "w") as hist_file:
            json.dump(self.hist, hist_file)

    def load_data(self):
        """
        Loads the current experiment history to append new results
        """
        with open(self.path, "r") as hist_file:
            self.hist = json.load(hist_file)

    def merge_logs(self, paths):
        print("Merge started...")
        for path in paths:
            print("Processing... %s" % path)
            with open(path) as hist_file:
                temp_hist = json.load(hist_file)

            for key in temp_hist:
                if key in self.hist:
                    self.hist[key].extend(temp_hist[key])
                else:
                    self.hist[key] = temp_hist[key]

        # Save the changes
        self.save_data()
        print("Merge completed.")
        print("New file is saved at %s" % self.path)

    def get_info_loss_mode(self, s):
        temp = 4 if s.find("info=") == -1 else 5
        i = max(s.find("info="), s.find("aug="))
        result = ""
        while s[i + temp] != "]":
            result += s[i + temp]
            i += 1

        return result

    def process_key(self, key):
        if key == "None":
            return "Baseline"

        return key

    def run_PACS_analysis(self, hist):
        pacs = [
            "PACS:Photo",
            "PACS:Art",
            "PACS:Cartoon",
            "PACS:Sketch"
        ]

        prefix = [
            "\\begin{table*}",
            "\t \\begin{adjustbox}{width=1\\textwidth}",
            "\t \\centering",
            "\t\t \\begin{tabular}{lcccccc} ",
            "\t\t \\toprule",
            "\t\t & Photo & Art & Cartoon & Sketch & Avg. & Max. \\\\",
            "\t\t \\midrule",
        ]

        # History contains repeated experiments, so compute mean & std
        hist_flag = False
        processed_hist = {}
        cumulative_hist = {}
        for i in hist:
            processed_hist[i] = {}
            cumulative_hist[i] = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
            for entry in hist[i]:
                for key in entry:
                    if key == "history": # Cumulative learning curve
                        for j in entry[key]:
                            cumulative_hist[i][j].append(entry[key][j])
                            hist_flag = True

                    else:
                        if key in processed_hist[i]:
                            processed_hist[i][key].append(entry[key])
                        else:
                            processed_hist[i][key] = [entry[key]]

        # Plot cumulative learning curve
        if hist_flag:
            for i in cumulative_hist:
                temp_hist = {}
                for key in cumulative_hist[i]:
                    temp = np.array(cumulative_hist[i][key])
                    temp_avg = np.mean(temp, axis=0)
                    temp_std = np.std(temp, axis=0)
                    temp_hist[key] = temp_avg
                    temp_hist["%s_std" % key] = temp_std
                plot_learning_curve(temp_hist, os.path.join(self.to_dir, "%s_PACS_learning_curve.%s" % (self.get_info_loss_mode(i), self.image_format)))

        # Prepare results in LaTeX format
        results = []
        for i in processed_hist:

            entry = processed_hist[i]
            temp_obj = "" if "vanilla" in i else " + contrastive"
            info_loss_mode = self.get_info_loss_mode(i)
            key = info_loss_mode + temp_obj
            key = key.replace("_", " ")
            key = self.process_key(key)

            accs = {}
            stds = {}
            for dataset in entry:
                cumulative_results = 100 * np.array(entry[dataset])
                avg = np.mean(cumulative_results)
                std = np.std(cumulative_results)
                accs[dataset] = float(avg)
                stds[dataset] = float(std)

            # Calculate avg performance per entry
            exp_count = len(entry["PACS:Photo"])
            avg_list = np.zeros(exp_count)
            for j in range(exp_count):
                avg_list[j] = np.mean(100 * np.array([entry[temp][j] for temp in entry if temp != "PACS:Photo" and temp in pacs]))

            # PACS table
            temp = "\t\t %s " % key
            while len(temp) < 52:
                temp += " "
            for j in pacs:
                temp += "& $%.2f \\pm %.1f$ " % (accs[j], stds[j]) if self.include_std else "& $%.2f$ " % (accs[j])
            avg, std, top = float(np.mean(avg_list)), float(np.std(avg_list)), float(avg_list.max())
            temp += "& $%.2f \\pm %.1f$ & $%.2f$ \\\\" % (avg, std, top)
            results.append(temp)

        body = ["\n".join(prefix)]
        body.append("\n".join(results))
        body.append("\t\t \\bottomrule")
        body.append("\t\t \\end{tabular}")
        body.append("\t \\end{adjustbox}")
        body.append("\t \\caption{ResNet-18 results on PACS benchmark}")
        body.append("\\end{table*}")

        # Export the LaTeX file
        with open(os.path.join(self.to_dir, "result_PACS.tex"), '+w') as tex_file:
            tex_file.write("\n".join(body))

    def run_COCO_analysis(self, hist):
        coco = [
            "COCO",
            "DomainNet:Real",
            "DomainNet:Painting",
            "DomainNet:Infograph",
            "DomainNet:Clipart",
            "DomainNet:Sketch",
            "DomainNet:Quickdraw"
        ]

        prefix = [
            "\\begin{table*}",
            "\t \\centering",
            #"\t\t \\begin{tabular}{lcccccccc} ",
            "\t\t \\begin{tabular}{lccccccccc} ",
            "\t\t \\toprule",
            #"\t\t & COCO & Real & Painting & Infograph & Clipart & Sketch & Avg. & Max. \\\\",
            "\t\t & COCO & Real & Painting & Infograph & Clipart & Sketch & Quickdraw & Avg. & Max. \\\\",
            "\t\t \\midrule",
        ]

        # History contains repeated experiments, so compute mean & std
        hist_flag = False
        processed_hist = {}
        cumulative_hist = {}
        for i in hist:
            processed_hist[i] = {}
            cumulative_hist[i] = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
            for entry in hist[i]:
                for key in entry:
                    if key == "history":  # Cumulative learning curve
                        for j in entry[key]:
                            cumulative_hist[i][j].append(entry[key][j])
                            hist_flag = True

                    else:
                        if key in processed_hist[i]:
                            processed_hist[i][key].append(entry[key])
                        else:
                            processed_hist[i][key] = [entry[key]]

        # Plot cumulative learning curve
        if hist_flag:
            for i in cumulative_hist:
                temp_hist = {}
                for key in cumulative_hist[i]:
                    temp = np.array(cumulative_hist[i][key])
                    temp_avg = np.mean(temp, axis=0)
                    temp_std = np.std(temp, axis=0)
                    temp_hist[key] = temp_avg
                    temp_hist["%s_std" % key] = temp_std
                plot_learning_curve(temp_hist, os.path.join(self.to_dir, "%s_COCO_learning_curve.%s" % (self.get_info_loss_mode(i), self.image_format)))

        # Prepare results in LaTeX format
        results = []
        for i in processed_hist:
            entry = processed_hist[i]
            temp_obj = "" if "vanilla" in i else " + contrastive"
            info_loss_mode = self.get_info_loss_mode(i)
            key = info_loss_mode + temp_obj
            key = key.replace("_", " ")
            key = self.process_key(key)

            accs = {}
            stds = {}
            for dataset in entry:
                cumulative_results = 100 * np.array(entry[dataset])
                avg = np.mean(cumulative_results)
                std = np.std(cumulative_results)
                accs[dataset] = float(avg)
                stds[dataset] = float(std)

            # Calculate avg performance per entry
            exp_count = len(entry["COCO"])
            avg_list = np.zeros(exp_count)
            for j in range(exp_count):
                avg_list[j] = np.mean(100 * np.array([entry[temp][j] for temp in entry if temp != "COCO" and temp in coco]))

            # COCO table
            temp = "\t\t %s " % key
            while len(temp) < 52:
                temp += " "
            for j in coco:
                temp += "& $%.2f \\pm %.1f$ " % (accs[j], stds[j]) if self.include_std else "& $%.2f$ " % (accs[j])
            avg, std, top = float(np.mean(avg_list)), float(np.std(avg_list)), float(avg_list.max())
            temp += "& $%.2f \\pm %.1f$ & $%.2f$ \\\\" % (avg, std, top)
            results.append(temp)

        body = ["\n".join(prefix)]
        body.append("\n".join(results))
        body.append("\t\t \\bottomrule")
        body.append("\t\t \\end{tabular}")
        body.append("\t \\vspace{-3pt}")
        body.append("\t \\caption{ResNet-18 results on COCO benchmark}")
        body.append("\\end{table*}")

        # Export the LaTeX file
        with open(os.path.join(self.to_dir, "result_COCO.tex"), '+w') as tex_file:
            tex_file.write("\n".join(body))

    def run_FullDomainNet_analysis(self, hist):
        domainnet = [
            "FullDomainNet:Real",
            "FullDomainNet:Painting",
            "FullDomainNet:Infograph",
            "FullDomainNet:Clipart",
            "FullDomainNet:Sketch",
            "FullDomainNet:Quickdraw"
        ]

        prefix = [
            "\\begin{table*}",
            "\t \\centering",
            "\t\t \\begin{tabular}{lcccccccc} ",
            "\t\t \\toprule",
            "\t\t & Real & Painting & Infograph & Clipart & Sketch & Quickdraw & Avg. & Max. \\\\",
            "\t\t \\midrule",
        ]

        # History contains repeated experiments, so compute mean & std
        hist_flag = False
        processed_hist = {}
        cumulative_hist = {}
        for i in hist:
            processed_hist[i] = {}
            cumulative_hist[i] = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
            for entry in hist[i]:
                for key in entry:
                    if key == "history":  # Cumulative learning curve
                        for j in entry[key]:
                            cumulative_hist[i][j].append(entry[key][j])
                            hist_flag = True

                    else:
                        if key in processed_hist[i]:
                            processed_hist[i][key].append(entry[key])
                        else:
                            processed_hist[i][key] = [entry[key]]

        # Plot cumulative learning curve
        if hist_flag:
            for i in cumulative_hist:
                temp_hist = {}
                for key in cumulative_hist[i]:
                    temp = np.array(cumulative_hist[i][key])
                    temp_avg = np.mean(temp, axis=0)
                    temp_std = np.std(temp, axis=0)
                    temp_hist[key] = temp_avg
                    temp_hist["%s_std" % key] = temp_std
                plot_learning_curve(temp_hist, os.path.join(self.to_dir, "%s_DomainNet_learning_curve.%s" % (self.get_info_loss_mode(i), self.image_format)))

        # Prepare results in LaTeX format
        results = []
        for i in processed_hist:
            entry = processed_hist[i]
            temp_obj = "" if "vanilla" in i else " + contrastive"
            info_loss_mode = self.get_info_loss_mode(i)
            key = info_loss_mode + temp_obj
            key = key.replace("_", " ")
            key = self.process_key(key)

            accs = {}
            stds = {}
            for dataset in entry:
                cumulative_results = 100 * np.array(entry[dataset])
                avg = np.mean(cumulative_results)
                std = np.std(cumulative_results)
                accs[dataset] = float(avg)
                stds[dataset] = float(std)

            # Calculate avg performance per entry
            exp_count = len(entry["FullDomainNet:Real"])
            avg_list = np.zeros(exp_count)
            for j in range(exp_count):
                avg_list[j] = np.mean(100 * np.array([entry[temp][j] for temp in entry if temp != "FullDomainNet:Real" and temp in domainnet]))

            # DomainNet table
            temp = "\t\t %s " % key
            while len(temp) < 52:
                temp += " "
            for j in domainnet:
                temp += "& $%.2f \\pm %.1f$ " % (accs[j], stds[j]) if self.include_std else "& $%.2f$ " % (accs[j])
            avg, std, top = float(np.mean(avg_list)), float(np.std(avg_list)), float(avg_list.max())
            temp += "& $%.2f \\pm %.1f$ & $%.2f$ \\\\" % (avg, std, top)
            results.append(temp)

        body = ["\n".join(prefix)]
        body.append("\n".join(results))
        body.append("\t\t \\bottomrule")
        body.append("\t\t \\end{tabular}")
        body.append("\t \\vspace{-3pt}")
        body.append("\t \\caption{ResNet-18 results on DomainNet benchmark}")
        body.append("\\end{table*}")

        # Export the LaTeX file
        with open(os.path.join(self.to_dir, "result_DomainNet.tex"), '+w') as tex_file:
            tex_file.write("\n".join(body))

    def run(self):
        benchmarks = {"PACS": {"hist": {}, "analysis_func": self.run_PACS_analysis},
                      "COCO": {"hist": {}, "analysis_func": self.run_COCO_analysis},
                      "FullDomainNet": {"hist": {}, "analysis_func": self.run_FullDomainNet_analysis}}

        for experiment in self.hist:

            if "PACS" in experiment:
                benchmarks["PACS"]["hist"][experiment] = self.hist[experiment]

            elif "COCO" in experiment:
                benchmarks["COCO"]["hist"][experiment] = self.hist[experiment]

            elif "FullDomainNet" in experiment:
                benchmarks["FullDomainNet"]["hist"][experiment] = self.hist[experiment]

        for benchmark in benchmarks:
            if len(benchmarks[benchmark]) > 0:
                benchmarks[benchmark]["analysis_func"](benchmarks[benchmark]["hist"])

if __name__ == '__main__':
    # Dynamic parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--path", default="../generalization.json", help="filepath to experiment history (JSON file)", type=str)
    parser.add_argument("--to_dir", default="../results", help="filepath to save charts, models, etc.", type=str)
    parser.add_argument("--merge_logs", help="merges experiment results for the given JSON file paths", nargs="+")
    parser.add_argument("--image_format", default="png", help="", type=str)
    args = vars(parser.parse_args())

    hist_path = args["path"]
    experimentProcessor = GeneralizationExpProcessor(config=args, path=hist_path, include_std=True, image_format=args["image_format"], to_dir=args["to_dir"])
    if args["merge_logs"] is None:
        experimentProcessor.run()
    else:
        experimentProcessor.merge_logs(paths=args["merge_logs"])