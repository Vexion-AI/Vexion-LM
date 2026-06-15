# Copyright 2026 Dmitry
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


import numpy as np
from tokenizers import Tokenizer
import os

TXT_FILE = "val.txt"     
BIN_FILE = "val.bin"     
TOKENIZER_PATH = "tokenizer.json" 

print("Загрузка токенизатора...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tokenizer.token_to_id("<|endoftext|>")

print(f"Начинаем конвертацию {TXT_FILE} в бинарный формат...")

with open(BIN_FILE, 'wb') as f_out:
    with open(TXT_FILE, 'r', encoding='utf-8') as f_in:
        batch_ids = []
        token_count = 0
        
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue
            
            ids = tokenizer.encode(line).ids + [eos_id]
            batch_ids.extend(ids)
            
            if len(batch_ids) >= 9_000_000:
                arr = np.array(batch_ids, dtype=np.uint16)
                f_out.write(arr.tobytes())
                token_count += len(batch_ids)
                batch_ids = []
                print(f"✅ Обраработано строк: {i:,} | Сохранено токенов в .bin: {token_count:,}")

        if len(batch_ids) > 0:
            arr = np.array(batch_ids, dtype=np.uint16)
            f_out.write(arr.tobytes())
            token_count += len(batch_ids)

print(f"🎉 Готово! Всего токенов в бинарнике: {token_count:,}")
