import json
import datasets


class HourVideoDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="HourVideo Dataset.",
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_path = "lmms_eval/tasks/hourvideo/dev_v1.0_annotations.json"
        return [
            datasets.SplitGenerator(name=datasets.NamedSplit("dev"), gen_kwargs={"filepath": data_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = []
            for video_id, video_data in data.items():
                questions.extend(video_data["benchmark_dataset"])
            for idx, question in enumerate(questions):
                yield idx, question
