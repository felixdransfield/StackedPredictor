from Models.Utils import get_distribution_scalars
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import json
from matplotlib import colors as mcolors


class ClassificationReport:
    def __init__(self):
        configs = json.load(open('Configuration.json', 'r'))

        self.model_results = []
        self.num_models=0
        self.output_path =   configs['paths']['classification_report_path']
        self.colors = mcolors.XKCD_COLORS

    def add_model_result( self, label, y_true, y_pred_binary, best_threshold,
                          precision_rt, recall_rt,yhat ):
        new_result = ModelResult(label, y_true, y_pred_binary, best_threshold, precision_rt,
                                 recall_rt,yhat)
        self.model_results.append(new_result)
        self.num_models = self.num_models+1

    def plot_distributions_vs_aucs( self ):
        percents = []
        pr_aucs = []
        roc_aucs = []
        outcomes = []
        print(" IN PLOT DISTRIBUTIONS, BEFORE FILTERING MODEL SUBSEST LENGTH: ",
              len(self.model_results))

        model_subsets = [x for x in self.model_results if ("3D" not in x.label) and ("5D" not in x.label)]
        print(" IN PLOT DISTRIBUTIONS, AFTER FILTERING MODEL SUBSEST LENGTH: ", len(model_subsets))
        for rs in model_subsets:
            outcome = rs.label
            minority_percent = rs.get_minority_percentage()
            pr_auc = rs.get_pr_auc()
            roc_auc = rs.get_roc_auc()

            outcomes.append(outcome)
            percents.append(minority_percent)
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)

        seq_len = range(1, len(outcomes)+1)
        fig, ax = plt.subplots()
        plt.figure(figsize=(15, 8))
        pecents_str = [str(round(x,2))+y for x,y in zip(percents,outcomes)]
        ax.set_xticklabels(pecents_str)

        sizes = [x*100 for x in percents]
        print(" length of sequence: ", len(seq_len))
        print(" length of precision and roc: ", len(pr_aucs), len(roc_aucs))
        rects1 = ax.scatter(seq_len, pr_aucs, s=sizes, label='PR AUC')
        rects2 = ax.scatter(seq_len, roc_aucs, s=sizes, label='ROC AUC')

        #plt.plot(percents, pr_aucs, label="PR AUC")
        #plt.plot(percents, roc_aucs, label="ROC AUC")

        fig.xlim([min(percents), max(percents)])
        fig.ylim([0, 1.01])
        fig.legend(loc='lower right')
        fig.title('')
        fig.xlabel('Minority Class %')
        fig.ylabel('Performance AUCs')
        fig.legend(loc='lower right')
        plt.xticks(percents, pecents_str, rotation='vertical')

        def autolabel ( rects ) :
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects :
                height = rect.get_height()
                plt.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()

        plt.savefig(self.output_path + "distribution_plot.pdf", bbox_inches='tight')

    def plot_pr_auc( self ):
        plt.figure(figsize=(15, 8))
        model_subsets = [x for x in self.model_results if ("3D" not in x.label)
                         and ("5D" not in x.label)]
        for rs in model_subsets:
            pr_auc = auc(rs.recall_vector, rs.precision_vector)

            if 'Mortality' in rs.label:
                style = 'dotted'
            else:
                style = 'dashed'

            plt.plot(rs.recall_vector, rs.precision_vector, linewidth=1.5,
                     linestyle=style, label=rs.label+' %0.3f' % pr_auc)
            plt.plot([0, 1], [1, 0], linewidth=1.5, linestyle='solid')

            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title(' Precision Recall Curve')
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        plt.savefig(self.output_path + "pr_auc.pdf", bbox_inches='tight')


    def plot_auc( self ):
        plt.figure(figsize=(15, 8))
        model_subsets = [x for x in self.model_results if ("3D" not in x.label)
                         and ("5D" not in x.label)]
        for rs in model_subsets :
            fpr, tpr, _ = roc_curve(rs.y_true, rs.y_pred)
            roc_auc = auc(fpr, tpr)
            if 'Mortality' in rs.label:
                style = 'dotted'
            else:
                style = 'dashed'

            plt.plot(fpr, tpr, linewidth=1.5, linestyle = style, label=rs.label+' %0.3f' % roc_auc)
            plt.plot([0, 0], [1, 1], linestyle = 'solid', linewidth=1.5)

            plt.xlim([-0.01, 1])
            plt.ylim([0, 1.01])
            plt.legend(loc='lower right')
            plt.title(' ROC Curve')
            plt.ylabel('False Positive Rate')
            plt.xlabel('True Positive Rate')

        plt.savefig(self.output_path + "auc.pdf", bbox_inches='tight')

class ModelResult:
    def __init__(self, label, y_true, y_pred_binary, best_threshold,
                 precision_rt, recall_rt,yhat ):
        self.label = label
        self.y_true = y_true
        self.y_pred = y_pred_binary
        self.threshold = best_threshold
        self.precision_vector = precision_rt
        self.recall_vector = recall_rt
        self.yhat = yhat

    def get_roc_auc (self):
        fpr, tpr, thresh = roc_curve(self.y_true, self.yhat)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_pr_auc (self):
        pr_auc =  auc(self.recall_vector, self.precision_vector)
        return pr_auc

    def get_minority_percentage( self ):
        distr = get_distribution_scalars(self.y_true)
        print(distr[1])
        return (distr[1])


