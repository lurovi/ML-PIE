import torch
import random


class TorchDataHandler:
    def __init__(self, dataset, batch_size, num_classes=0, labels_names=None, transform=None, shuffle=False):
        if len(dataset) % batch_size != 0:
            raise ValueError("Length of dataset / batch size must be an integer.")
        self.curr_batch = 0
        self.num_classes = num_classes
        self.number_of_records = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = int(len(dataset)/batch_size)
        self.transform = transform
        self.original_dataset = {i: dataset[i][0] for i in range(self.number_of_records)}
        self.labels = {i: dataset[i][1] for i in range(self.number_of_records)}
        if not( self.transform ):
            self.transform = lambda val: val
        self.data = [(i, self.transform(dataset[i][0]), dataset[i][1]) for i in range(self.number_of_records)]
        if not( labels_names ):
            self.labels_names = {i: str(i) for i in range(self.num_classes)}
        else:
            self.labels_names = labels_names

    def get_num_classes(self):
        return self.num_classes

    def get_label_name(self, label_id):
        return self.labels_names[label_id]

    def get_num_batches(self):
        return self.num_batches

    def __len__(self):
        return self.number_of_records

    def get_original_sample(self, idx):
        return self.original_dataset[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_record(self, idx):
        return self.transform(self.get_original_sample(idx))

    def __getitem__(self, idx):
        label = self.get_label(idx)
        original = self.get_original_sample(idx)
        record = self.get_record(idx)
        return {"id": idx, "label": label, "category": self.get_label_name(label),
                "original": original, "record": record}

    def enable_shuffle(self):
        self.shuffle = True

    def disable_shuffle(self):
        self.shuffle = False

    def next_batch(self):
        if self.curr_batch == 0 & self.shuffle:
            random.shuffle(self.data)
        start_ind = self.curr_batch * self.batch_size  # included
        end_ind = start_ind + self.batch_size  # not included
        curr_data = self.data[start_ind:end_ind]
        self.curr_batch = (self.curr_batch+1) % self.num_batches
        n = []
        ids = []
        objs = []
        labels = []
        for h in curr_data:
            ids.append(h[0])
            objs.append(h[1].tolist())
            labels.append(h[2])
        ids = torch.tensor(ids)
        objs = torch.tensor(objs)
        labels = torch.tensor(labels)
        n.append(ids)
        n.append(objs)
        n.append(labels)
        return n

    def all_batches(self):
        if self.curr_batch != 0:
            raise ValueError('The method all_batches can be executed if and only if the current batch begins with the '
                             'first one.')
        n = []
        for i in range(self.get_num_batches()):
            n.append(self.next_batch())
        return n
