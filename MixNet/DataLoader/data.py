from typing import Union
from torch.utils.data import Dataset, DataLoader
from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.interfaces import IDataLoader, IPipeline, IPadder

SPK_ID = 0
PATH_ID = 1
TEXT_ID = 2
DURATION_ID = 3


class Data(Dataset):
    def __init__(
            self,
            data_loader: IDataLoader,
            aud_pipeline: IPipeline,
            text_pipeline: IPipeline,
            aud_padder: IPadder,
            text_padder: IPadder,
            batch_size: int,
            sep: str
            ) -> None:
        super().__init__()
        self.sep = sep
        self.aud_pipeline = aud_pipeline
        self.text_pipeline = text_pipeline
        self.aud_padder = aud_padder
        self.text_padder = text_padder
        self.batch_size = batch_size
        self.data = self.process(data_loader)
        self.max_speech_lens = []
        self.max_text_lens = []
        self.n_batches = len(self.data) // self.batch_size
        if len(self.data) % batch_size > 0:
            self.n_batches += 1
        self.__set_max_text_lens()

    def process(self, data_loader: IDataLoader):
        data = data_loader.load().split('\n')
        data = [item.split(self.sep) for item in data]
        #data=sorted(data, key=itemgetter(DURATION_ID))
        data = sorted(data, key=lambda x: float(x[DURATION_ID]), reverse=True)
        return data

    def __set_max_text_lens(self):
        for i, item in enumerate(self.data):
            idx = i // self.batch_size
            length = len(item[TEXT_ID])
            if idx >= len(self.max_text_lens):
                self.max_text_lens.append(length)
            else:
                self.max_text_lens[idx] = max(length, self.max_text_lens[idx])

    def __len__(self) -> int:
        return len(self.data)

    def _get_max_len(self, idx: int) -> Union[None, int]:
        bucket_id = idx // self.batch_size
        if bucket_id >= len(self.max_speech_lens):
            return None, self.max_text_lens[bucket_id] + 1
        return (
            self.max_speech_lens[bucket_id],
            self.max_text_lens[bucket_id] + 1
            )

    def __getitem__(self, idx: int):
        [embeds_path, file_path, text, _] = self.data[idx]
        return embeds_path, file_path, text


def get_batch_loader(
        data_loader: IDataLoader,
        aud_pipeline: IPipeline,
        text_pipeline: IPipeline,
        aud_padder: IPadder,
        text_padder: IPadder,
        batch_size: int,
        sep: str
        ):
    return DataLoader(
        Data(
            data_loader=data_loader,
            aud_pipeline=aud_pipeline,
            text_pipeline=text_pipeline,
            aud_padder=aud_padder,
            text_padder=text_padder,
            batch_size=batch_size,
            sep=sep
        ),
        batch_size=batch_size,
        shuffle=False
    )

