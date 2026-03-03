import csv
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class ChunkInfo:
    size_bytes: int
    duration: float
    bitrate_kbps: int

class BigBuckBunnyDataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0 # action 인데스
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration

class sintelDataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f) 
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration


class ToSDataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f) 
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration

class Swiss_Dataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f) 
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration

class Elephants_Dataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f) 
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration

class Forest_men_Dataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.chunk_table: Dict[int, Dict[int, ChunkInfo]] = defaultdict(dict)
        self.num_chunks: int = 0
        self.num_reps: int = 0
        self.load()
    
    def load(self):
        with self.csv_path.open() as f:
            reader = csv.DictReader(f) 
            for row in reader:
                chunk = int(row["chunk_idx"])
                rep = int(row["rep_idx"])
                size = int(row["size_bytes"])
                dur = float(row["duration_sec"])
                br = int(row["bitrate_kbps"])

                self.chunk_table[chunk][rep] = ChunkInfo(size, dur, br)
                self.num_chunks = max(self.num_chunks, chunk)
                self.num_reps = max(self.num_reps, rep + 1)
    
    def get_chunk_size(self, chunk_idx: int, rep_idx: int):
        return self.chunk_table[chunk_idx][rep_idx].size_bytes
    
    def get_chunk_duration(self, chunk_idx: int):
        return next(iter(self.chunk_table[chunk_idx].values())).duration