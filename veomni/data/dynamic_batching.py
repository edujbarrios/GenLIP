# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import sys
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, Optional

from ..utils import logging

import traceback
import threading
from queue import Queue
from threading import Thread


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from .batching_strategy import BaseBatchingStrategy

from .multimodal.data_collator import my_collate_fn

class DynamicBatchSizeDataLoaderold:
    """Dynamic batch DataLoader.

    Args:
        dataloader: torch DataLoader
        batching_strategy: dynamic batch strategy
        collate_fn: DataLoader collate_fn, collate data after get data from batching_strategy
        num_micro_batch: num_micro_batch, if num_micro_batch == 1, return micro_batch for gradient accumulation
        length: length of dataloader, if length == -1, length = sys.maxsize, default len(dataloader)
        drop_last: if True, drop last batch if batch size < num_micro_batch

    """

    def __init__(
        self,
        dataloader: Any,
        batching_strategy: "BaseBatchingStrategy",
        collate_fn: Optional[Callable] = None,
        micro_batch_size: int = 1,
        num_micro_batch: int = 1,
        length: int = 0,
        drop_last: bool = True,
        max_seq_len: int = 0,
    ) -> None:
        self.batching_strategy = batching_strategy
        self.micro_batch_size = micro_batch_size
        self.num_micro_batch = num_micro_batch
        self.dataloader_item_buffer = deque()
        self.item_buffer = deque()
        self.step = 0
        self._collate_fn = collate_fn
        self._dataloader = dataloader
        self._drop_last = drop_last
        self._data_iter: Iterator
        self._resume = False
        self._batch_data_iter: Generator
        self.max_seq_len = max_seq_len

        if length > 0:
            self._length = length
        elif length == -1:
            self._length = sys.maxsize
        else:
            self._length = len(self._dataloader)

    def __len__(self):
        if self._length:
            return self._length
        else:
            raise RuntimeError("length must set at init. before call len()")

    def __iter__(self) -> Iterator:
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
            self._batch_data_iter = self.batch_data_generator()
        self._resume = False
        return self

    def __next__(self):
        return next(self._batch_data_iter)

    def batch_data_generator(self):
        batch = []
        micro_batchs = []

        while True:
            if self._length and self.step >= self._length:
                return

            if self.batching_strategy.is_full_filled():
                micro_batch = self.batching_strategy.get_micro_batch(self.step)
                if self._collate_fn:
                    micro_batch = self._collate_fn(micro_batch)
                micro_batchs.append(micro_batch)
                if len(micro_batchs) == self.micro_batch_size:
                    batch.append(my_collate_fn(micro_batchs, self.max_seq_len))
                    micro_batchs = []
                    if len(batch) == self.num_micro_batch:
                        yield batch
                        self.step += 1
                        batch = []
            
            # if self.batching_strategy.is_full_filled():
            #     micro_batch = self.batching_strategy.get_micro_batch(self.step)
            #     if self._collate_fn:
            #         micro_batch = self._collate_fn(micro_batch)
            #     batch.append(micro_batch)
            #     if len(batch) == self.num_micro_batch:
            #         yield batch
            #         self.step += 1
            #         batch = []


            try:
                processing_item = next(self._data_iter)
            except Exception as e:
                if isinstance(e, StopIteration):
                    if self.step < self._length:
                        # call iter until reach length
                        self._data_iter = iter(self._dataloader)
                        processing_item = next(self._data_iter)
                    elif not self._drop_last and not self.batching_strategy.empty():
                        while not self.batching_strategy.empty():
                            micro_batch = self.batching_strategy.get_micro_batch(self.step)
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            batch.append(micro_batch)
                            if len(batch) == self.num_micro_batch:
                                yield batch
                                self.step += 1
                                batch = []

                        while len(batch) < self.num_micro_batch:
                            padding_batch = copy.deepcopy(micro_batch)
                            padding_batch["padding_flag"] = True
                            batch.append(padding_batch)
                        yield batch
                        self.step += 1
                        return
                    else:
                        return
                else:
                    logger.error(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise

            # if processing_item is None:
            #     continue

            # put processing_item to buffer
            if isinstance(processing_item, dict):
                processing_item = [processing_item]

            for item in processing_item:
                if item is None:
                    print(f"item is None, skip")
                    continue
                self.batching_strategy.put_item(item[0])

    def state_dict(self):
        # save state
        state = self.__dict__.copy()
        # remove internal fields
        for k in list(state.keys()):
            if k.startswith("_"):
                del state[k]

        # save dataloader state
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()

        if hasattr(self.batching_strategy, "state_dict"):
            state["batching_strategy_state"] = self.batching_strategy.state_dict()  # type: ignore
            del state["batching_strategy"]

        return copy.deepcopy(state)

    def load_state_dict(self, state: Dict[str, Any]):
        if state["num_micro_batch"] != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state['num_micro_batch']} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
            del state["num_micro_batch"]
        self.__dict__.update(state)
        self._resume = True

        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])

        if "batching_strategy_state" in state:
            self.batching_strategy.load_state_dict(  # type: ignore
                state["batching_strategy_state"]
            )
            del state["batching_strategy_state"]

        self._data_iter = iter(self._dataloader)
        self._batch_data_iter = self.batch_data_generator()


class DynamicBatchSizeDataLoader:
    """Dynamic batch DataLoader.
    Args:
        dataloader: torch DataLoader
        batching_strategy: dynamic batch strategy
        collate_fn: DataLoader collate_fn, collate data after get data from batching_strategy
        micro_batch_size: The size of each micro-batch.
        num_micro_batch: The number of micro-batches to form a full batch.
        length: length of dataloader, if length == -1, length = sys.maxsize, default len(dataloader)
        drop_last: if True, drop last batch if batch size < num_micro_batch
        prefetch_factor (int): Number of batches to prefetch in the background. Defaults to 2.
                               Set to 0 to disable prefetching.
        max_seq_len (int): Maximum sequence length for padding.
    """
    def __init__(
        self,
        dataloader: Any,
        batching_strategy: "BaseBatchingStrategy",
        collate_fn: Optional[Callable] = None,
        micro_batch_size: int = 1,
        num_micro_batch: int = 1,
        length: int = 0,
        drop_last: bool = True,
        # --- NEW PARAMETER ---
        prefetch_factor: int = 4,
        max_seq_len: int = 0,
    ) -> None:
        self.batching_strategy = batching_strategy
        self.micro_batch_size = micro_batch_size
        self.num_micro_batch = num_micro_batch
        self.step = 0
        self._collate_fn = collate_fn
        self._dataloader = dataloader
        self._drop_last = drop_last
        self.max_seq_len = max_seq_len
        # --- NEW PREFETCH ATTRIBUTES ---
        self.prefetch_factor = prefetch_factor
        self._prefetch_queue: Optional[Queue] = None
        self._prefetch_thread: Optional[Thread] = None
        self._sentinel = object() # A unique object to signal the end of iteration
        self._data_iter: Iterator
        self._resume = False
        if length > 0:
            self._length = length
        elif length == -1:
            self._length = sys.maxsize
        else:
            self._length = len(self._dataloader)
    def __len__(self):
        if self._length:
            return self._length
        else:
            raise RuntimeError("length must set at init. before call len()")
    # --- MODIFIED __iter__ to handle prefetching ---
    def __iter__(self) -> Iterator:
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
        # If prefetching is enabled, set up the queue and background thread
        if self.prefetch_factor > 0:
            self._prefetch_queue = Queue(maxsize=self.prefetch_factor)
            self._prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
            self._prefetch_thread.start()
        # If prefetching is disabled, fall back to the original generator
        else:
            self._batch_data_iter = self.batch_data_generator()
        self._resume = False
        return self
    # --- MODIFIED __next__ to pull from the prefetch queue ---
    def __next__(self):
        # If prefetching, get from the queue
        if self.prefetch_factor > 0 and self._prefetch_queue is not None:
            batch = self._prefetch_queue.get()
            # Check for the sentinel value to know when to stop
            if batch is self._sentinel:
                # Clean up the thread
                if self._prefetch_thread is not None:
                    self._prefetch_thread.join()
                raise StopIteration
            return batch
        # Otherwise, use the original generator
        else:
            return next(self._batch_data_iter)
    # --- NEW METHOD: The loop that runs in the background thread ---
    def _prefetch_loop(self):
        """
        This method runs in a background thread. It generates batches
        and puts them into the prefetch queue.
        """
        try:
            for batch in self.batch_data_generator():
                if self._prefetch_queue is not None:
                    self._prefetch_queue.put(batch)
        except Exception as e:
            logger.error(f"Exception in prefetch loop: {e}\n{traceback.format_exc()}")
            # Put the exception into the queue so the main thread can raise it
            if self._prefetch_queue is not None:
                self._prefetch_queue.put(e)
        finally:
            # When the generator is exhausted, put the sentinel value in the queue
            # to signal the main thread that there are no more items.
            if self._prefetch_queue is not None:
                self._prefetch_queue.put(self._sentinel)
    def batch_data_generator(self) -> Generator:
        batch = []
        micro_batchs = []
        while True:
            if self._length and self.step >= self._length:
                return
            if self.batching_strategy.is_full_filled():
                micro_batch = self.batching_strategy.get_micro_batch(self.step)
                if self._collate_fn:
                    micro_batch = self._collate_fn(micro_batch)
                micro_batchs.append(micro_batch)
                if len(micro_batchs) == self.micro_batch_size:
                    batch.append(my_collate_fn(micro_batchs, self.max_seq_len))
                    micro_batchs = []
                    if len(batch) == self.num_micro_batch:
                        yield batch
                        self.step += 1
                        batch = []
            try:
                processing_item = next(self._data_iter)
            except Exception as e:
                if isinstance(e, StopIteration):
                    if self.step < self._length:
                        self._data_iter = iter(self._dataloader)
                        processing_item = next(self._data_iter)
                    elif not self._drop_last and not self.batching_strategy.empty():
                        # This part handles the final, potentially incomplete batch
                        while not self.batching_strategy.empty():
                            micro_batch = self.batching_strategy.get_micro_batch(self.step)
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            batch.append(micro_batch)
                            if len(batch) == self.num_micro_batch:
                                yield batch
                                self.step += 1
                                batch = []
                        if batch: # If there's a leftover partial batch
                            while len(batch) < self.num_micro_batch:
                                padding_batch = copy.deepcopy(batch[-1]) # Use last valid micro-batch for padding structure
                                padding_batch["padding_flag"] = True
                                batch.append(padding_batch)
                            yield batch
                            self.step += 1
                        return
                    else:
                        return
                else:
                    logger.error(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise
            if isinstance(processing_item, dict):
                processing_item = [processing_item]
            for item in processing_item:
                if item is None:
                    print(f"item is None, skip")
                    continue
                self.batching_strategy.put_item(item[0])
    # --- MODIFIED state_dict and load_state_dict to handle prefetch ---
    def state_dict(self):
        state = self.__dict__.copy()
        for k in list(state.keys()):
            # Remove internal and prefetch-related fields that shouldn't be saved
            if k.startswith("_") or k in ['_prefetch_queue', '_prefetch_thread', '_sentinel']:
                del state[k]
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()
        if hasattr(self.batching_strategy, "state_dict"):
            state["batching_strategy_state"] = self.batching_strategy.state_dict()
            del state["batching_strategy"]
        return copy.deepcopy(state)
    def load_state_dict(self, state: Dict[str, Any]):
        if state.get("num_micro_batch") != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state.get('num_micro_batch')} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
        self.__dict__.update(state)
        self._resume = True
        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])
        if "batching_strategy_state" in state:
            self.batching_strategy.load_state_dict(state["batching_strategy_state"])
        # Important: Do not start the prefetch thread here.
        # __iter__ will be called by the training loop, which will then correctly
        # initialize the data iterator and start the prefetch thread.
        self._data_iter = iter(self._dataloader)
        # When resuming, we don't pre-initialize the generator or thread.
        # The next call to `iter(dataloader)` will handle it.



class DynamicBatchSizeDataLoadernew:
    """Dynamic batch DataLoader with prefetch.
    Args:
        dataloader: torch DataLoader
        batching_strategy: dynamic batch strategy
        collate_fn: DataLoader collate_fn, collate data after get data from batching_strategy
        micro_batch_size: number of micro-batches that form one collated micro-batch via my_collate_fn
        num_micro_batch: number of collated micro-batches that form one training step batch
        length: length of dataloader; if -1, use sys.maxsize; default len(dataloader)
        drop_last: if True, drop last batch if batch size < num_micro_batch
        max_seq_len: passed to my_collate_fn
        prefetch_factor: number of step-batches to pre-build (producer fills this many ahead)
    """
    def __init__(
        self,
        dataloader: Any,
        batching_strategy: "BaseBatchingStrategy",
        collate_fn: Optional[Callable] = None,
        micro_batch_size: int = 1,
        num_micro_batch: int = 1,
        length: int = 0,
        drop_last: bool = True,
        max_seq_len: int = 4,
        prefetch_factor: int = 0,  # NEW
    ) -> None:
        self.batching_strategy = batching_strategy
        self.micro_batch_size = micro_batch_size
        self.num_micro_batch = num_micro_batch
        self.dataloader_item_buffer = deque()
        self.item_buffer = deque()
        self.step = 0
        self._collate_fn = collate_fn
        self._dataloader = dataloader
        self._drop_last = drop_last
        self._data_iter: Iterator
        self._resume = False
        self._batch_data_iter: Generator
        self.max_seq_len = max_seq_len
        # length handling
        if length > 0:
            self._length = length
        elif length == -1:
            self._length = sys.maxsize
        else:
            self._length = len(self._dataloader)
        # Prefetch-related
        self.prefetch_factor = max(0, int(prefetch_factor))
        self._prefetch_buffer: deque = deque()
        self._prefetch_lock = threading.Lock()
        self._prefetch_not_empty = threading.Condition(self._prefetch_lock)
        self._prefetch_not_full = threading.Condition(self._prefetch_lock)
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_producer = threading.Event()
        self._producer_started = False
        self._producer_exception: Optional[BaseException] = None
    def __len__(self):
        if self._length:
            return self._length
        else:
            raise RuntimeError("length must set at init. before call len()")
    def __iter__(self) -> Iterator:
        # Stop any previous producer if resuming/starting a new iteration
        self._stop_and_join_producer()
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
            # The generator is still used by the producer internally
            self._batch_data_iter = self._batch_data_generator()
        self._resume = False
        # Start producer if prefetch > 0, else we will directly consume from the generator
        if self.prefetch_factor > 0:
            self._start_producer()
        return self
    def __next__(self):
        # If using prefetch, consume from buffer; otherwise pull directly from generator
        if self.prefetch_factor <= 0:
            return next(self._batch_data_iter)
        # Check if producer raised
        self._maybe_raise_producer_error()
        with self._prefetch_lock:
            # Wait until buffer not empty or producer finished
            while len(self._prefetch_buffer) == 0 and self._producer_thread is not None and self._producer_thread.is_alive():
                self._prefetch_not_empty.wait(timeout=0.5)
            # Check again, either we have data, or producer ended
            if len(self._prefetch_buffer) == 0:
                # No data available. Producer might be done or errored.
                self._maybe_raise_producer_error()
                # Producer finished and no data left -> StopIteration
                self._stop_and_join_producer()
                raise StopIteration
            batch = self._prefetch_buffer.popleft()
            # Signal producer there is room
            self._prefetch_not_full.notify()
            return batch
    def _start_producer(self):
        if self._producer_started:
            return
        self._producer_exception = None
        self._stop_producer.clear()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="DynamicBatchDataLoaderProducer",
            daemon=True,
        )
        self._producer_thread.start()
        self._producer_started = True
    def _stop_and_join_producer(self):
        if self._producer_thread is not None:
            self._stop_producer.set()
            # Wake any waits
            with self._prefetch_lock:
                self._prefetch_not_full.notify_all()
                self._prefetch_not_empty.notify_all()
            self._producer_thread.join(timeout=1.0)
        self._producer_thread = None
        self._producer_started = False
        self._stop_producer.clear()
        # Do not clear buffer here; iteration end will naturally drain.
    def _maybe_raise_producer_error(self):
        if self._producer_exception is not None:
            exc = self._producer_exception
            self._producer_exception = None
            # Ensure thread is stopped
            self._stop_and_join_producer()
            raise exc
    def _producer_loop(self):
        try:
            # Produce until generator finishes or stop requested
            while not self._stop_producer.is_set():
                # Pull next batch
                try:
                    next_batch = next(self._batch_data_iter)
                except StopIteration:
                    # End of data
                    break
                # Put into buffer, blocking if full
                with self._prefetch_lock:
                    while len(self._prefetch_buffer) >= self.prefetch_factor and not self._stop_producer.is_set():
                        self._prefetch_not_full.wait(timeout=0.2)
                    if self._stop_producer.is_set():
                        break
                    self._prefetch_buffer.append(next_batch)
                    self._prefetch_not_empty.notify()
        except BaseException as e:
            # Save exception to raise on consumer side
            self._producer_exception = e
            logger.error(f"Producer encountered exception: {e}\n{traceback.format_exc()}")
            # Wake consumer
            with self._prefetch_lock:
                self._prefetch_not_empty.notify_all()
        finally:
            # Mark end: no special marker, consumer checks thread state + buffer emptiness
            pass
    def _batch_data_generator(self):
        # Kept as a private method; public interface consumes via __next__
        batch = []
        micro_batchs = []
        while True:
            if self._length and self.step >= self._length:
                return
            if self.batching_strategy.is_full_filled():
                micro_batch = self.batching_strategy.get_micro_batch(self.step)
                if self._collate_fn:
                    micro_batch = self._collate_fn(micro_batch)
                micro_batchs.append(micro_batch)
                if len(micro_batchs) == self.micro_batch_size:
                    # my_collate_fn expected to collate a list of micro-batches into one
                    collated_micro = my_collate_fn(micro_batchs, self.max_seq_len)
                    micro_batchs = []
                    batch.append(collated_micro)
                    if len(batch) == self.num_micro_batch:
                        yield batch
                        self.step += 1
                        batch = []
            # Fetch next raw item to feed batching_strategy
            try:
                processing_item = next(self._data_iter)
            except Exception as e:
                if isinstance(e, StopIteration):
                    if self.step < self._length:
                        # recycle underlying dataloader until logical length reached
                        self._data_iter = iter(self._dataloader)
                        processing_item = next(self._data_iter)
                    elif not self._drop_last and not self.batching_strategy.empty():
                        # Flush remaining micro-batches
                        while not self.batching_strategy.empty():
                            micro_batch = self.batching_strategy.get_micro_batch(self.step)
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            micro_batchs.append(micro_batch)
                            if len(micro_batchs) == self.micro_batch_size:
                                collated_micro = my_collate_fn(micro_batchs, self.max_seq_len)
                                micro_batchs = []
                                batch.append(collated_micro)
                                if len(batch) == self.num_micro_batch:
                                    yield batch
                                    self.step += 1
                                    batch = []
                        # If we still have partial collated micro(s), collate and pad batch to num_micro_batch
                        if len(micro_batchs) > 0:
                            collated_micro = my_collate_fn(micro_batchs, self.max_seq_len)
                            batch.append(collated_micro)
                            micro_batchs = []
                        if len(batch) > 0:
                            last_valid = batch[-1]
                            while len(batch) < self.num_micro_batch:
                                padding_batch = copy.deepcopy(last_valid)
                                if isinstance(padding_batch, dict):
                                    padding_batch["padding_flag"] = True
                                else:
                                    # If not dict, fallback attr
                                    try:
                                        padding_batch.padding_flag = True  # type: ignore
                                    except Exception:
                                        pass
                                batch.append(padding_batch)
                            yield batch
                            self.step += 1
                        return
                    else:
                        return
                else:
                    logger.error(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise
            # Normalize processing_item to list of dicts
            if isinstance(processing_item, dict):
                processing_item = [processing_item]
            for item in processing_item:
                if item is None:
                    print(f"item is None, skip")
                    continue
                # Original code used item[0]; preserve that behavior if items are tuples/lists
                feed = item[0] if isinstance(item, (list, tuple)) else item
                self.batching_strategy.put_item(feed)
    def state_dict(self):
        # Stop producer to avoid races while serializing
        self._stop_and_join_producer()
        # save state
        state = self.__dict__.copy()
        # remove internal fields (those starting with _) and runtime-only objects
        for k in list(state.keys()):
            if k.startswith("_"):
                del state[k]
        # Explicitly remove thread/locks if any slipped through
        state.pop("dataloader_item_buffer", None)  # these are user-visible but safe to serialize if needed
        state.pop("item_buffer", None)
        # save dataloader state
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()
        if hasattr(self.batching_strategy, "state_dict"):
            state["batching_strategy_state"] = self.batching_strategy.state_dict()  # type: ignore
            # do not serialize the strategy object itself
            del state["batching_strategy"]
        return copy.deepcopy(state)
    def load_state_dict(self, state: Dict[str, Any]):
        # Stop producer to avoid races when mutating state
        self._stop_and_join_producer()
        if state["num_micro_batch"] != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state['num_micro_batch']} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
            del state["num_micro_batch"]
        # prefetch_factor can change across runs; allow overwrite if present in state
        self.__dict__.update(state)
        self._resume = True
        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])
        if "batching_strategy_state" in state:
            self.batching_strategy.load_state_dict(  # type: ignore
                state["batching_strategy_state"]
            )
            del state["batching_strategy_state"]
        # Reset runtime objects
        self._data_iter = iter(self._dataloader)
        self._batch_data_iter = self._batch_data_generator()
        # Clear prefetch buffers and runtime flags
        with self._prefetch_lock:
            self._prefetch_buffer.clear()
        self._producer_exception = None
        self._producer_started = False
        self._stop_producer.clear()