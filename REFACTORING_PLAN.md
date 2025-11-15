# CosyVoice2AudioDecoder 重構計劃

## 📊 當前狀態分析

### 已完成 (Phase 1-3)
- ✅ **Phase 1**: 移除不可達代碼 (40 行)
- ✅ **Phase 2**: 提取魔術數字為類常數
- ✅ **Phase 3** (3.1-3.12): 統一變數命名規範
  - Batch/Token/Mask/Hidden/Time/Length/Prompt/Sample/Speech/Text/Attention 變數
  - 移除冗餘變數 (t_starts, token_end_list)
  - 移除冗餘參數 (device_target, is_streaming 等)

### 方法長度分析
| 方法名稱 | 行數 | 狀態 | 優先級 |
|---------|------|------|--------|
| `flow()` | **982** | ⚠️ 超長 | 🔴 **高** |
| `stream_inference_with_text()` | **339** | ⚠️ 超長 | 🔴 **高** |
| `__init__()` | 271 | ⚡ 較長 | 🟡 中 |
| `stream_inference()` | 242 | ⚡ 較長 | 🟡 中 |
| `_maybe_warm_start_flows()` | 164 | ⚡ 較長 | 🟢 低 |

---

## 🎯 Phase 4: 重構 `flow()` 方法

### 目標
將 `flow()` 從 982 行縮減至 ~200 行，提升嵌套函數為類方法

### 4.1 提升嵌套輔助函數為類私有方法 (14 個函數)

#### 需提升的函數清單：

**設備與輸入處理**:
1. `_infer_device_from_batch(batch_dict, fallback_device)` → `self._infer_device_from_batch(...)`
2. `_prepare_inputs_and_ensure_device(batch_dict, target_device)` → `self._prepare_inputs_and_ensure_device(...)`

**文本上下文構建**:
3. `_build_cross_attention_text_context(...)` → `self._build_cross_attention_text_context(...)`
4. `_build_fallback_text_context(...)` → `self._build_fallback_text_context(...)`

**非流式路徑**:
5. `_process_nonstreaming_path(...)` → `self._process_nonstreaming_path(...)`

**條件構建與損失計算**:
6. `_build_condition_and_compute_loss(...)` → `self._build_condition_and_compute_loss(...)`
7. `_print_training_debug_info(...)` → `self._print_training_debug_info(...)`

**流式專用 - 樣本提取與預計算**:
8. `_extract_sample_tokens_and_text(...)` → `self._extract_sample_tokens_and_text(...)`
9. `_precompute_token_and_text_embeddings(...)` → `self._precompute_token_and_text_embeddings(...)`

**流式專用 - 分塊處理**:
10. `_compute_first_chunk_length(...)` → `self._compute_first_chunk_length(...)`
11. `_build_chunk_boundaries(...)` → `self._build_chunk_boundaries(...)`
12. `_slice_and_batch_token_embeddings(...)` → `self._slice_and_batch_token_embeddings(...)`

**流式專用 - 文本與時間軸重建**:
13. `_build_text_context_for_chunks(...)` → `self._build_text_context_for_chunks(...)`
14. `_reconstruct_sample_timeline(...)` → `self._reconstruct_sample_timeline(...)`

**流式專用 - Debug 與批次處理**:
15. `_print_microbatch_debug_info(...)` → `self._print_microbatch_debug_info(...)`
16. `_pad_and_concatenate_batch(...)` → `self._pad_and_concatenate_batch(...)`

### 4.2 檢查並移除冗餘參數

提升後需檢查的方向：
- `device_target` 是否可改用 `self.device` 或從 tensor 推斷？
- 重複傳遞的配置（如 `upsample_f`, `block_size`）是否可作為方法內部獲取？
- `b_idx` (batch index) 僅用於 debug，是否可簡化？

### 4.3 簡化 `flow()` 主邏輯

**重構前** (982 行):
```python
def flow(self, batch, device):
    # 定義 16 個嵌套函數 (700+ 行)
    def _helper1(...): ...
    def _helper2(...): ...
    # ...
    
    # 主邏輯 (200+ 行)
    streaming = ...
    if streaming:
        # 複雜流式邏輯
    else:
        # 非流式邏輯
    return {'loss': loss}
```

**重構後** (目標 ~200 行):
```python
def flow(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """訓練損失計算主入口"""
    # 1. 設備與輸入準備
    device = self._infer_device_from_batch(batch, device)
    inputs = self._prepare_inputs_and_ensure_device(batch, device)
    
    # 2. 決定訓練模式
    streaming = self._should_use_streaming_training()
    
    # 3. 編碼
    if streaming:
        hidden, hidden_mask = self._encode_streaming(inputs)
    else:
        hidden, hidden_mask = self._encode_nonstreaming(inputs)
    
    # 4. 損失計算
    loss = self._compute_training_loss(hidden, hidden_mask, inputs)
    
    # 5. Debug 輸出
    self._log_training_metrics(loss, streaming, inputs)
    
    self._step += 1
    return {'loss': loss}
```

---

## 🎯 Phase 5: 重構 `stream_inference_with_text()`

### 目標
從 339 行縮減至 ~150 行，提取重複邏輯

### 5.1 提取 Cross-Attention 邏輯

**識別到的重複模式**:
- `stream_inference_with_text()` 中的 cross-attention (L2135-2191)
- `flow()` 中的 `_build_cross_attention_text_context()` (L820-998)

**建議**: 統一為 `self._build_streaming_text_context(token_hist, text_tokens, text_end_idx, context_len)`

### 5.2 提取共用代碼

**Mel Overlap 處理** (在 3 處重複):
- `stream_inference()` L1947-1962
- `stream_inference_with_text()` L2282-2297
- 邏輯完全相同

**建議**: 提取為 `self._apply_mel_overlap(new_mel, prev_mel, finalize)`

**Batch 拆分邏輯** (在 3 處重複):
- `stream_inference()` L1836-1856
- `stream_inference_with_text()` L2082-2103
- `offline_inference()` L2408-2440

**建議**: 提取為 `self._process_batch(token, uuid, prompt_token, ...)`

### 5.3 統一 Upsampling 邏輯

**重複模式**:
```python
# 在 4 處重複
if upsample_factor > 1:
    token_upsampled = token_win.repeat_interleave(upsample_factor, dim=1)
    prompt_token_upsampled = prompt_token_hist.repeat_interleave(...)
    # ...
```

**建議**: 提取為 `self._upsample_tokens(token, prompt_token, upsample_factor)`

---

## 🎯 Phase 7 (審計): 整理 `__init__()` 配置

### 目標
整理 271 行的 `__init__()`，為配置添加分類註釋

### 配置變數分類 (共 20+ 個)

#### 1. 訓練配置
```python
self._stream_train_prob: float           # 流式訓練概率
self._use_text_context_train: bool       # 訓練時使用文本上下文
self._stream_train_first_block_random: bool  # 隨機首塊長度
self._fixed_window_pad: bool             # 固定窗口填充
```

#### 2. Cross-Attention 配置
```python
self._use_cross_text_attn: bool          # 啟用跨文本注意力
self._cross_text_heads: int              # 注意力頭數
self._cross_text_dropout: float          # Dropout 率
self._cross_q_pool: str                  # Query 池化方式
self._cross_ffn_hidden: int              # FFN 隱藏層大小
self._cross_ffn_dropout: float           # FFN Dropout
```

#### 3. Speech/Text Self-Attention 配置
```python
self._speech_sa_heads: int               # Speech self-attn 頭數
self._speech_sa_dropout: float
self._speech_ffn_hidden: int
self._text_sa_heads: int                 # Text self-attn 頭數
self._text_sa_dropout: float
self._text_ffn_hidden: int
```

#### 4. 流式推理配置
```python
self._stream_stride: int                 # 滑動窗口步長
self._token_overlap_len: int             # Token 重疊長度
self._mel_overlap_len: int               # Mel 重疊長度
```

#### 5. Debug 配置
```python
self._step: int                          # 當前步數
self._debug_every: int                   # Debug 頻率
self._val_debug: bool                    # 驗證時 debug
self._debug_text_align: bool             # Text/token 對齊 debug
self._debug_xattn: bool                  # Cross-attention debug
self._debug_blocks: bool                 # Block 統計 debug
self._print_per_n_chunk: int             # 打印頻率
self._print_every_k_in_batch: int        # 批內打印間隔
```

#### 6. 模型結構配置
```python
self._default_context_len: int           # 上下文長度
self._ctx_dim: int                       # 上下文維度
self._text_vocab_size: int               # 文本詞表大小
```

#### 7. 其他
```python
self._cos2_config_override: Dict         # CosyVoice2 配置覆蓋
self._warm_cfg: Dict                     # 熱啟動配置
self._fallback_dir: str                  # 回退模型路徑
```

**建議改動**:
1. 在 `__init__()` 中添加區塊註釋分隔各類配置
2. 按類別重新排序參數和賦值
3. 統一從環境變數/YAML/參數獲取配置的優先級邏輯

---

## 🎯 Phase 9: 改進類型註解

### 目標
為所有方法添加完整的類型提示

### 9.1 公有方法 (必須完整)

**當前狀態**:
```python
# ✅ 已完整
def flow(self, batch: Dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:

# ❌ 缺失返回值詳細類型
@torch.inference_mode()
def token2wav(self, token, uuid, prompt_token=..., ...) -> tuple[torch.Tensor, torch.Tensor]:
```

**建議改進**:
```python
from typing import Dict, Optional, Tuple, List

@torch.inference_mode()
def token2wav(
    self,
    token: torch.Tensor,
    uuid: str,
    prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int64),
    prompt_feat: Optional[torch.Tensor] = None,
    embedding: Optional[torch.Tensor] = None,
    finalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tts_speech: Waveform at 22050 Hz [B, T_wav]
        tts_mel: Generated mel spectrogram [B, 80, T_mel]
    """
```

### 9.2 私有方法 (至少返回值類型)

**改進案例**:
```python
# 當前
def _determine_chunk_block_size(self):
    block_size = ...
    return block_size

# 改為
def _determine_chunk_block_size(self) -> int:
    """Determine chunking block size in original token units."""
    block_size = ...
    return block_size
```

---

## 🎯 Phase 10: 統一錯誤處理

### 目標
改進 20+ 處 `try-except-pass` 反模式

### 問題模式

**案例 1: 靜默失敗**:
```python
# L1180 (出現 10+ 次)
try:
    logging.info(...)
except Exception:
    pass  # ❌ 完全靜默，難以調試
```

**案例 2: 過於寬泛**:
```python
# L886-889
try:
    _m_pre = float(_s_pre.mean().item())
    _sd_pre = float(_s_pre.std(unbiased=False).item())
except Exception:  # ❌ 捕獲所有異常
    _m_pre = 0.0; _sd_pre = 0.0
```

### 改進方案

**選項 1: 具體異常 + Debug 日誌**:
```python
try:
    logging.info(f"[cos2.train] loss={loss:.6f}")
except (AttributeError, RuntimeError) as e:
    if self._val_debug:
        logging.debug(f"Failed to log training metric: {e}")
```

**選項 2: 輔助函數**:
```python
def _safe_log_metric(self, msg: str, level: str = "info"):
    """Safely log metrics without crashing on formatting errors."""
    try:
        getattr(logging, level)(msg)
    except Exception as e:
        if self._val_debug:
            logging.debug(f"Logging failed: {e}")

# 使用
self._safe_log_metric(f"[cos2.train] loss={loss:.6f}")
```

**選項 3: 裝飾器 (Phase 10.2)**:
```python
def safe_debug_log(func):
    """Decorator to safely execute debug logging functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.debug(f"{func.__name__} failed: {e}")
            return None
    return wrapper

@safe_debug_log
def _log_training_metrics(self, loss, streaming, inputs):
    logging.info(f"[cos2.train] ...")
```

---

## 📋 執行順序與優先級

### 第一階段 (本週)
1. ✅ **Phase 4.1**: 提升 `flow()` 嵌套函數 (預計 2-3 小時)
2. ✅ **Phase 4.2**: 檢查冗餘參數 (預計 1 小時)
3. ✅ **Phase 4.3**: 簡化 `flow()` 主邏輯 (預計 1-2 小時)

### 第二階段 (下週)
4. ✅ **Phase 7**: 整理 `__init__()` 配置註釋 (預計 1 小時)
5. ✅ **Phase 5.1-5.3**: 重構 `stream_inference_with_text()` (預計 3-4 小時)

### 第三階段 (後續)
6. ✅ **Phase 9.1-9.2**: 添加類型註解 (預計 2-3 小時)
7. 🟡 **Phase 10.1-10.2**: 改進錯誤處理 (預計 2 小時，可選)

---

## 📊 預期成果

| 指標 | 當前 | Phase 4-5 完成後 | 改善幅度 |
|-----|------|-----------------|---------|
| `flow()` 行數 | 982 | ~200 | **↓ 80%** |
| `stream_inference_with_text()` 行數 | 339 | ~150 | **↓ 56%** |
| 最長方法行數 | 982 | ~250 | **↓ 75%** |
| 嵌套函數數量 | 16 | 0 | **↓ 100%** |
| 類方法數量 | 11 | ~25 | **↑ 127%** |
| 代碼重複處 | ~15 處 | ~5 處 | **↓ 67%** |

---

## 🎯 完成標準

### Phase 4 完成標準
- [ ] `flow()` 中所有嵌套函數已提升為類方法
- [ ] `flow()` 主邏輯 ≤ 250 行
- [ ] 所有提升的方法都有 docstring
- [ ] 通過現有的訓練測試
- [ ] 無新增 linter 錯誤

### Phase 5 完成標準
- [ ] `stream_inference_with_text()` ≤ 200 行
- [ ] Mel overlap 邏輯統一為單一方法
- [ ] Batch 處理邏輯統一
- [ ] 通過推理測試

### Phase 7 完成標準
- [ ] `__init__()` 配置按類別分組並註釋
- [ ] 配置獲取邏輯統一且清晰

### Phase 9 完成標準
- [ ] 所有公有方法有完整類型註解
- [ ] 所有私有方法有返回值類型註解
- [ ] IDE 自動補全正常工作

---

## 🚫 不執行的項目

- ❌ **Phase 6**: 參數對象化 (subfunctions 參數多是合理的)
- ❌ **Phase 7**: 提取配置類 (會影響外部調用方式)
- ❌ **Phase 8**: 模組化拆分 (維護成本高，風險大)

