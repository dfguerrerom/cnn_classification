import pandas as pd
from box import Box
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from config import conf


class Writer:

    METRICS = Box(
        {
            "loss_train": {
                "row": 1,
                "col": 1,
            },
            "loss_val": {
                "row": 1,
                "col": 1,
            },
            "accuracy": {
                "row": 1,
                "col": 2,
            },
            "lr": {
                "row": 1,
                "col": 3,
            },
        }
    )

    def __init__(self):

        self.data_df = pd.DataFrame(
            data={
                "metric": [],
                "epoch": [],
                "value": [],
            }
        )

        self.model_name = None

    def update(self, metric, val, epoch):
        """Adds each final epoch metric to the writer dataframe"""

        # TODO: Not only store the last epoch value,
        # we could also add each batch progress, and so display the info
        # every single batch improve.
        # Metrics has to be calculated in the epoch loop

        if metric not in self.METRICS:
            raise ValueError(f"Only {self.METRICS.keys()} are accepted.")

        new_data_df = pd.DataFrame(
            data={
                "metric": [metric],
                "epoch": [epoch],
                "value": [val],
            }
        )
        self.data_df = pd.concat([self.data_df, new_data_df])

    def __str__(self):
        """Calls summary and print metrics from the last epoch"""
        return self.data_df

    def summary(self):
        """Create a summary output and replace previously created one"""

    def save(self, model_name, metadata=None):
        """Save current status of the dataframe"""

        self.file = conf.out_history / f"{self.model_name}.csv"
        if metadata:
            with open(
                self.file.parent / (self.file.name + "_metadata.json"), "w"
            ) as out:
                out.write(metadata)

        self.data_df.to_csv(self.file)

    def plot(self):
        """Create subplots to display and update each metric graphs"""

        self.fig = go.FigureWidget(
            make_subplots(rows=1, cols=3, subplot_titles=("Loss", "Accuracy", "LR"))
        )

        [
            self.fig.add_trace(go.Scatter(name=k), row=v.row, col=v.col)
            for k, v in self.METRICS.items()
        ]

        return self.fig

    def update_plot(self):
        """Update each subplot based on the current dataframe status"""

        self.fig.update_layout(title_text=self.model_name)

        def get_metric(metric):
            return self.data_df[self.data_df.metric == metric].sort_values(by=["epoch"])

        [
            self.fig.update_traces(
                x=get_metric(metric)["epoch"],
                y=get_metric(metric)["value"],
                selector=dict(name=metric),
            )
            for metric in self.METRICS
        ]

    def last_metric(self, metric_name):
        """Returns last metric (last epoch) value from the data dataframe"""

        return (
            self.data_df[self.data_df.metric == metric_name]
            .sort_values(by=["epoch"], ascending=False)["value"]
            .iloc[0]
        )

    def load_data(self, metrics_file):
        """loads the content of the metrics and load them in the graphs"""

        self.data_df = pd.read_csv(metrics_file)
        self.plot()
        self.update_plot()

        return self.fig
