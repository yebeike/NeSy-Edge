from transformers import AutoTokenizer, AutoConfig
import json

# 刚才下载的那个模型
model_id = "rd211/Qwen3-0.6B-Instruct"

print(f"🕵️‍♀️ 正在尸检模型: {model_id}...\n")

try:
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # 2. 检查特殊 Token (是不是混入了 <think>?)
    print("="*40)
    print("1. 特殊 Token 映射 (Special Tokens Map)")
    print("="*40)
    for k, v in tokenizer.special_tokens_map.items():
        print(f"{k}: {v}")
    
    print(f"\nAdditional Special Tokens: {tokenizer.additional_special_tokens}")
    
    # 专门搜查 <think>
    vocab = tokenizer.get_vocab()
    suspects = ["<think>", "<|thought|>", "<thought>"]
    print("\n🔍 搜查嫌疑 Token:")
    for s in suspects:
        if s in vocab:
            print(f"⚠️ 发现嫌疑犯 '{s}': ID = {vocab[s]}")
        else:
            print(f"✅ 未发现 '{s}'")

    # 3. 检查 Chat Template (最关键的证据)
    # 这里定义了 user/assistant/system 怎么拼接。
    # 如果这里面写死了 "<think>"，那就是实锤了。
    print("\n" + "="*40)
    print("2. 聊天模板 (Chat Template)")
    print("="*40)
    if tokenizer.chat_template:
        print(tokenizer.chat_template)
    else:
        print("❌ 没有定义 Chat Template (这也很奇怪)")

    # 4. 检查生成配置
    print("\n" + "="*40)
    print("3. 默认生成配置")
    print("="*40)
    # 看看有没有强制的 eos_token_id 或者 bad_words_ids
    print(f"EOS Token ID: {config.eos_token_id}")
    print(f"BOS Token ID: {config.bos_token_id}")

except Exception as e:
    print(f"❌ 尸检失败: {e}")