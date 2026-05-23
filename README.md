# AI-Runtime-Recommendation-Deployment-System
本项目实现本地大模型运行时智能调度系统，解决开发者本地部署AI时的模型选择、硬件适配与多后端管理问题。支持Ollama与LM Studio双后端统一调度，自动检测GPU、Apple Silicon及CPU环境，基于显存、内存、磁盘空间进行模型兼容性分析与推荐。内置多维模型评分机制，结合HumanEval、MMLU、上下文长度、推理速度、中文能力等指标，针对编程、写作、推理、对话等场景动态加权排序。支持benchmark与真实token/s性能测试，持续优化推荐结果。提供Web UI、CLI、PyQt GUI三端交互，实现模型自动拉取、运行与生命周期管理。已完成本地AI环境初始化、大模型自动部署、多Runtime统一调度及低配置设备适配。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
异构本地大模型运行时与智能调度系统
================================================

依赖安装：
    pip install pynvml psutil aiohttp gradio requests PyQt6

运行示例：
    python main.py --purpose 编程
    python main.py --web
    python main.py --gui
    python main.py --list
"""

import os
import sys
import json
import asyncio
import shutil
import platform
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import psutil
import aiohttp
import pynvml

# =========================
# logging 配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("llm-installer")

# =========================
# 常量与配置
# =========================
BASE_DIR = Path(__file__).parent
MODELS_JSON = BASE_DIR / "models.json"
BENCHMARK_CACHE = BASE_DIR / "benchmark_cache.json"
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "models_remote_url": "https://raw.githubusercontent.com/your-org/llm-assistant/main/models.json",
    "auto_benchmark": True
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()

# =========================
# 数据结构
# =========================
@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_gb: float
    disk_free_gb: float
    has_nvidia: bool
    has_apple_silicon: bool
    is_cpu_only: bool

    @property
    def effective_vram_gb(self) -> float:
        if self.has_apple_silicon:
            return self.ram_gb * 0.75
        if self.is_cpu_only:
            return 1.0
        return self.vram_free_gb

# =========================
# 模型数据库在线更新
# =========================
def update_models_from_remote():
    config = load_config()
    remote_url = config.get("models_remote_url", "")
    if not remote_url:
        logger.error("未配置 models_remote_url，请在 config.json 中设置")
        return False
    try:
        import requests
        logger.info(f"正在从 {remote_url} 下载模型数据库...")
        resp = requests.get(remote_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            with open(MODELS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("模型数据库更新成功")
            return True
        else:
            logger.error("远程数据格式无效")
            return False
    except Exception as e:
        logger.error(f"更新失败: {e}")
        return False

def load_models() -> List[Dict]:
    if not MODELS_JSON.exists():
        raise FileNotFoundError(f"模型数据库不存在: {MODELS_JSON}")
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# 硬件检测（Web/CLI/GUI 兼容）
# =========================
def _is_web_mode() -> bool:
    return "--web" in sys.argv

def _is_gui_mode() -> bool:
    return "--gui" in sys.argv

def detect_hardware() -> HardwareProfile:
    has_nvidia = False
    has_apple_silicon = False
    is_cpu_only = False
    gpu_name = "CPU only"
    vram_total_gb = 0.0
    vram_free_gb = 0.0

    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total_gb = mem.total / (1024**3)
            vram_free_gb = mem.free / (1024**3)
            has_nvidia = True
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug(f"NVIDIA 检测失败: {e}")

    if not has_nvidia and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            brand = result.stdout.strip()
            if "Apple" in brand:
                has_apple_silicon = True
                gpu_name = brand
                logger.info(f"检测到 Apple Silicon: {brand}")
        except Exception:
            pass

    # 无 GPU 时的交互处理（GUI/Web 模式下不阻塞）
    if not has_nvidia and not has_apple_silicon:
        logger.warning("未检测到独立 GPU")
        if _is_web_mode() or _is_gui_mode():
            user_vram = 0.0
            if _is_web_mode():
                print("\nWeb 模式下自动切换为纯 CPU 模式。")
        else:
            print("\n当前设备没有检测到 NVIDIA 或 Apple GPU。")
            while True:
                try:
                    user_vram = float(input("请输入可用显存 (GB, 纯CPU则输入0): ").strip() or "0")
                    if user_vram < 0:
                        raise ValueError
                    break
                except ValueError:
                    print("请输入数字（≥0）")
        vram_total_gb = user_vram
        vram_free_gb = user_vram
        if user_vram <= 1.0:
            is_cpu_only = True
            gpu_name = "CPU only (无有效GPU)"
            logger.info("运行模式: 纯 CPU（速度较慢）")
        else:
            gpu_name = f"User specified GPU ({user_vram}GB)"

    ram_gb = psutil.virtual_memory().total / (1024**3)
    models_path = Path.home() / ".ollama" / "models"
    exist_path = models_path
    while not exist_path.exists():
        exist_path = exist_path.parent
    disk_free_gb = psutil.disk_usage(str(exist_path)).free / (1024**3)

    logger.info(f"硬件: GPU={gpu_name} | 可用显存={vram_free_gb:.1f}GB | RAM={ram_gb:.1f}GB | 磁盘={disk_free_gb:.1f}GB")
    if is_cpu_only and not (_is_web_mode() or _is_gui_mode()):
        print("\n⚠️  注意：当前为纯 CPU 模式，推荐模型将限制为极小参数量（≤3B），运行速度可能较慢。")

    return HardwareProfile(
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        ram_gb=ram_gb,
        disk_free_gb=disk_free_gb,
        has_nvidia=has_nvidia,
        has_apple_silicon=has_apple_silicon,
        is_cpu_only=is_cpu_only
    )

# =========================
# 推荐系统 (通用)
# =========================
PURPOSE_WEIGHTS = {
    "编程": {"coding": 1.0, "chinese": 0.3, "reasoning": 0.8},
    "对话": {"coding": 0.1, "chinese": 1.0, "reasoning": 0.7},
    "写作": {"coding": 0.1, "chinese": 0.9, "reasoning": 0.7},
    "翻译": {"coding": 0.0, "chinese": 1.0, "reasoning": 0.4},
    "通用": {"coding": 0.4, "chinese": 0.8, "reasoning": 0.8},
}

def resolve_purpose(user_input: str) -> str:
    lower = user_input.lower()
    for key in PURPOSE_WEIGHTS:
        if key in lower:
            return key
    return "通用"

def model_score(model: Dict, hw: HardwareProfile, purpose: str, prefer_speed: bool = False) -> float:
    weights = PURPOSE_WEIGHTS[purpose]
    cap = model.get("capability", {})
    bench = model.get("benchmark", {"humaneval": 50, "mmlu": 50})

    # 对于 LM Studio 模型可能没有详细能力评分，使用默认值
    coding = cap.get("coding", 0.5)
    chinese = cap.get("chinese", 0.5)
    reasoning = cap.get("reasoning", 0.5)
    max_context = cap.get("max_context", 4096)
    speed_tps = cap.get("speed_tps", 20)
    min_vram = model.get("min_vram_gb", hw.effective_vram_gb * 0.8)

    capability = (weights["coding"] * coding +
                  weights["chinese"] * chinese +
                  weights["reasoning"] * reasoning)

    if purpose == "编程":
        bench_score = (bench.get("humaneval", 50) / 100) * 0.7 + (bench.get("mmlu", 50) / 100) * 0.3
    else:
        bench_score = (bench.get("mmlu", 50) / 100)

    if hw.is_cpu_only:
        fit_ratio = min(2.0 / max(min_vram, 0.1), 1.0)
    else:
        fit_ratio = min_vram / hw.effective_vram_gb
    fit_score = max(0.0, 1.0 - abs(1.0 - fit_ratio))

    context_bonus = min(max_context / 128000, 1.0) * 0.1
    speed_score = min(speed_tps / 100, 1.0) * 0.1

    final = (capability * 0.45 + bench_score * 0.30 + fit_score * 0.10 +
             speed_score * 0.10 + context_bonus)
    if prefer_speed:
        final += speed_score * 0.15
    if hw.is_cpu_only and min_vram > 3:
        final *= 0.5
    return final

def recommend_models(hw: HardwareProfile, models: List[Dict],
                     purpose: str, top_k: int = 3,
                     prefer_speed: bool = False) -> Tuple[List[Dict], bool]:
    purpose_key = resolve_purpose(purpose)
    compatible = []
    for m in models:
        min_vram = m.get("min_vram_gb", hw.effective_vram_gb * 0.8)
        min_ram = m.get("min_ram_gb", 4)
        disk_usage = m.get("disk_usage_gb", 2)
        if not hw.is_cpu_only:
            if min_vram > hw.effective_vram_gb - 0.5:
                continue
        else:
            if min_vram > 3.0:
                continue
        if min_ram > hw.ram_gb:
            continue
        if disk_usage > hw.disk_free_gb - 1.5:
            continue
        compatible.append(m)

    fallback = False
    if not compatible:
        logger.warning("没有完全满足硬件要求的模型，将展示资源需求最低的 3 个")
        compatible = sorted(models, key=lambda x: x.get("min_vram_gb", 999))[:3]
        fallback = True

    scored = [(m, model_score(m, hw, purpose_key, prefer_speed)) for m in compatible]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:top_k]], fallback

# =========================
# 真实性能测试（仅 Ollama）
# =========================
async def run_benchmark(model_tag: str) -> Optional[float]:
    logger.info(f"正在对模型 {model_tag} 进行性能测试...")
    prompt = "Explain quantum computing in one sentence."
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "run", model_tag,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        start = time.perf_counter()
        stdout, stderr = await proc.communicate(input=prompt.encode())
        end = time.perf_counter()
        if proc.returncode != 0:
            logger.error(f"性能测试失败: {stderr.decode()}")
            return None
        output_text = stdout.decode().strip()
        estimated_tokens = len(output_text.split()) * 1.3
        elapsed = end - start
        tps = estimated_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"性能测试结果: {tps:.1f} token/s (估算)")
        return tps
    except Exception as e:
        logger.error(f"性能测试异常: {e}")
        return None

async def benchmark_and_cache(model_tag: str) -> float:
    cache = {}
    if BENCHMARK_CACHE.exists():
        with open(BENCHMARK_CACHE, "r") as f:
            cache = json.load(f)
    if model_tag in cache:
        entry = cache[model_tag]
        if time.time() - entry.get("timestamp", 0) < 7 * 86400:
            logger.info(f"使用缓存的性能数据: {entry['tps']:.1f} token/s")
            return entry["tps"]
    tps = await run_benchmark(model_tag)
    if tps is not None:
        cache[model_tag] = {"tps": tps, "timestamp": time.time()}
        with open(BENCHMARK_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
        return tps
    else:
        models = load_models()
        for m in models:
            if m.get("ollama_tag") == model_tag:
                logger.warning(f"性能测试失败，使用模型默认速度 {m['capability']['speed_tps']} token/s")
                return m["capability"]["speed_tps"]
        return 10.0

# =========================
# Ollama 后端
# =========================
def check_ollama_installed() -> bool:
    return shutil.which("ollama") is not None

async def check_ollama_api() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
                return resp.status == 200
    except Exception:
        return False

async def ensure_ollama_running():
    if await check_ollama_api():
        return
    logger.info("启动 Ollama 服务...")
    if platform.system().lower() == "windows":
        asyncio.create_task(asyncio.create_subprocess_exec("ollama", "serve"))
    else:
        asyncio.create_task(asyncio.create_subprocess_exec(
            "ollama", "serve",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        ))
    await asyncio.sleep(3)
    if not await check_ollama_api():
        raise RuntimeError("无法启动 Ollama 服务，请手动执行 'ollama serve'")

async def install_ollama_windows():
    if shutil.which("winget"):
        logger.info("正在使用 winget 安装 Ollama...")
        proc = await asyncio.create_subprocess_exec(
            "winget", "install", "Ollama.Ollama", "--silent",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"winget 安装失败: {stderr.decode()}")
            return False
        logger.info("Ollama 安装完成，请重新启动终端或手动刷新环境变量")
        return True
    else:
        logger.warning("未找到 winget，请手动安装 Ollama")
        print("\n请访问 https://ollama.com/download 下载安装包，安装后重新运行。")
        return False

async def ensure_ollama_installed():
    if check_ollama_installed():
        return True
    logger.warning("Ollama 未安装，正在尝试自动安装...")
    system = platform.system().lower()
    if system == "windows":
        success = await install_ollama_windows()
        if success:
            await asyncio.sleep(2)
            return check_ollama_installed()
        return False
    else:
        proc = await asyncio.create_subprocess_shell(
            'curl -fsSL https://ollama.com/install.sh | sh',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        return check_ollama_installed()

async def get_ollama_models() -> List[str]:
    proc = await asyncio.create_subprocess_exec(
        "ollama", "list",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    installed = []
    for line in stdout.decode().splitlines()[1:]:
        if line.strip():
            installed.append(line.split()[0])
    return installed

async def pull_model_with_retry(model_tag: str, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"拉取模型: {model_tag} (尝试 {attempt}/{max_retries})")
            proc = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    if '%' in text or 'MB' in text:
                        print(f"\r  {text}", end="", flush=True)
                    else:
                        print(f"  {text}")
            print()
            await proc.wait()
            if proc.returncode == 0:
                logger.info(f"模型 {model_tag} 下载完成")
                return
            else:
                raise RuntimeError(f"拉取失败，退出码 {proc.returncode}")
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.error(f"尝试 {attempt} 失败: {e}")
            if attempt < max_retries:
                wait = 5 * attempt
                logger.info(f"等待 {wait} 秒后重试...")
                await asyncio.sleep(wait)
            else:
                logger.error("已达到最大重试次数，拉取失败")
                print(f"\n请手动执行: ollama pull {model_tag}")
                raise

async def run_ollama_interactive(model_tag: str):
    logger.info(f"启动模型: {model_tag}")
    subprocess.run(["ollama", "run", model_tag])

# =========================
# LM Studio 后端
# =========================
async def check_lm_studio_api() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:1234/v1/models", timeout=3) as resp:
                return resp.status == 200
    except Exception:
        return False

async def get_lm_studio_models() -> List[Dict]:
    """从 LM Studio API 获取已安装模型列表，并转换为统一的模型字典格式"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:1234/v1/models") as resp:
                data = await resp.json()
                models = []
                for item in data.get("data", []):
                    model_id = item["id"]
                    # 尝试从模型 ID 推断参数量（简单启发式）
                    # 例如 "qwen2.5-7b-instruct" -> 7B
                    size_gb = 4.0  # 默认值
                    if "7b" in model_id.lower():
                        size_gb = 5.0
                    elif "13b" in model_id.lower():
                        size_gb = 9.0
                    elif "3b" in model_id.lower():
                        size_gb = 2.5
                    elif "1b" in model_id.lower():
                        size_gb = 1.2
                    models.append({
                        "display_name": model_id,
                        "ollama_tag": model_id,  # 用 ID 代替 tag
                        "min_vram_gb": size_gb,
                        "min_ram_gb": 8,
                        "disk_usage_gb": size_gb,  # 实际大小从 API 无法获取
                        "capability": {
                            "coding": 0.6,
                            "chinese": 0.7,
                            "reasoning": 0.7,
                            "max_context": 4096,
                            "speed_tps": 25
                        },
                        "benchmark": {"humaneval": 50, "mmlu": 50},
                        "backend": "lmstudio"  # 标识为 LM Studio 模型
                    })
                return models
    except Exception as e:
        logger.error(f"获取 LM Studio 模型列表失败: {e}")
        return []

async def run_lm_studio_chat(model_id: str):
    """简单的终端聊天循环，通过 LM Studio API 交互"""
    print(f"\n启动与 {model_id} 的对话（输入 /exit 退出）")
    async with aiohttp.ClientSession() as session:
        messages = []
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip() == "/exit":
                    break
                messages.append({"role": "user", "content": user_input})
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.7,
                    "stream": False
                }
                async with session.post("http://127.0.0.1:1234/v1/chat/completions", json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        reply = data["choices"][0]["message"]["content"]
                        print(f"AI: {reply}")
                        messages.append({"role": "assistant", "content": reply})
                    else:
                        print(f"请求失败: {resp.status}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"聊天出错: {e}")

# =========================
# 统一后端选择与操作
# =========================
class BackendManager:
    def __init__(self, backend: str = "ollama"):
        self.backend = backend  # "ollama" 或 "lmstudio"

    async def is_available(self) -> bool:
        if self.backend == "ollama":
            return await check_ollama_api()
        else:
            return await check_lm_studio_api()

    async def ensure_running(self):
        if self.backend == "ollama":
            if not await check_ollama_api():
                if not check_ollama_installed():
                    raise RuntimeError("Ollama 未安装")
                await ensure_ollama_running()
        else:  # lmstudio
            if not await check_lm_studio_api():
                raise RuntimeError("LM Studio 服务未运行，请启动 LM Studio 并加载模型后重试")

    async def get_installed_models(self) -> List[Dict]:
        if self.backend == "ollama":
            tags = await get_ollama_models()
            # 从 models.json 中查找详细信息
            full_models = load_models()
            result = []
            for tag in tags:
                found = False
                for m in full_models:
                    if m["ollama_tag"] == tag:
                        result.append(m)
                        found = True
                        break
                if not found:
                    # 创建一个基础条目
                    result.append({
                        "display_name": tag,
                        "ollama_tag": tag,
                        "min_vram_gb": 4,
                        "min_ram_gb": 8,
                        "disk_usage_gb": 4,
                        "capability": {"coding": 0.5, "chinese": 0.7, "reasoning": 0.6, "max_context": 4096, "speed_tps": 30},
                        "benchmark": {"humaneval": 50, "mmlu": 50}
                    })
            return result
        else:
            return await get_lm_studio_models()

    async def pull_model(self, model_tag: str):
        if self.backend == "ollama":
            await pull_model_with_retry(model_tag)
        else:
            logger.info("LM Studio 模型请通过 LM Studio 界面下载")

    async def run_model(self, model_tag: str):
        if self.backend == "ollama":
            await run_ollama_interactive(model_tag)
        else:
            await run_lm_studio_chat(model_tag)

# =========================
# 输出显示
# =========================
def print_hardware(hw: HardwareProfile):
    sep = "─" * 60
    print(sep)
    print(f"  GPU / 加速器 : {hw.gpu_name}")
    if hw.has_apple_silicon:
        print(f"  统一内存     : {hw.ram_gb:.1f} GB  （可用显存估算 {hw.effective_vram_gb:.1f} GB）")
    elif hw.is_cpu_only:
        print(f"  模式         : 纯 CPU")
        print(f"  系统内存     : {hw.ram_gb:.1f} GB")
    else:
        print(f"  显存（空闲） : {hw.vram_free_gb:.1f} GB")
        print(f"  系统内存     : {hw.ram_gb:.1f} GB")
    print(f"  模型目录可用 : {hw.disk_free_gb:.1f} GB")
    print(sep)

def print_recommendations(ranked: List[Dict], fallback: bool, is_cpu_only: bool):
    if fallback:
        print("\n⚠️  警告：以下模型不完全满足硬件要求，运行可能缓慢或崩溃\n")
    if is_cpu_only:
        print("\n💡 提示：当前为纯 CPU 模式，以下模型均为轻量级（≤3B），速度可能较慢。")
    print(f"\n{'='*60}")
    print(f"  推荐模型 Top-{len(ranked)}")
    print(f"{'='*60}")
    for idx, m in enumerate(ranked, 1):
        cap = m.get("capability", {})
        bm = m.get("benchmark", {})
        tag = m.get("ollama_tag", m.get("display_name"))
        print(f"\n✨ [{idx}] {m['display_name']}  (tag: {tag})")
        print(f"      所需显存: {m.get('min_vram_gb', '?')} GB | 内存: {m.get('min_ram_gb', '?')} GB")
        print(f"      能力: 编程 {cap.get('coding', 0):.2f} | 中文 {cap.get('chinese', 0):.2f} | 推理 {cap.get('reasoning', 0):.2f}")
        print(f"      上下文: {cap.get('max_context', '?')} | 速度: {cap.get('speed_tps', '?')} token/s")
        if bm:
            print(f"      基准: HumanEval {bm.get('humaneval','N/A')} | MMLU {bm.get('mmlu','N/A')}")
    print("─" * 60)

def print_all_models(hw: HardwareProfile, models: List[Dict]):
    eff = hw.effective_vram_gb
    print(f"\n{'模型名称':<30} {'显存需求':>8} {'可运行':>6}")
    print("─" * 50)
    for m in sorted(models, key=lambda x: x.get("min_vram_gb", 0)):
        if hw.is_cpu_only:
            runnable = (m.get("min_vram_gb", 0) <= 3.0 and
                        m.get("min_ram_gb", 0) <= hw.ram_gb and
                        m.get("disk_usage_gb", 0) <= hw.disk_free_gb - 1.5)
        else:
            runnable = (m.get("min_vram_gb", 0) <= eff - 0.5 and
                        m.get("min_ram_gb", 0) <= hw.ram_gb and
                        m.get("disk_usage_gb", 0) <= hw.disk_free_gb - 1.5)
        flag = "✔" if runnable else "✘"
        print(f"{m.get('display_name', m.get('ollama_tag', '')):<30} {m.get('min_vram_gb', 0):>6.1f} GB  {flag:>4}")
    print()

# =========================
# Web UI 模块（保持兼容）
# =========================
def serve_web_ui():
    try:
        import gradio as gr
    except ImportError:
        logger.error("请先安装 gradio: pip install gradio")
        sys.exit(1)

    hw = detect_hardware()
    models = load_models()
    backend_mgr = BackendManager("ollama")  # Web 默认用 Ollama

    def recommend_ui(purpose: str):
        ranked, fallback = recommend_models(hw, models, purpose, top_k=5)
        md = f"### 硬件信息\n- {hw.gpu_name}\n"
        if hw.has_apple_silicon:
            md += f"- 统一内存: {hw.ram_gb:.1f} GB（可用显存估算 {hw.effective_vram_gb:.1f} GB）\n"
        elif hw.is_cpu_only:
            md += f"- 模式: 纯 CPU\n- 系统内存: {hw.ram_gb:.1f} GB\n"
        else:
            md += f"- 可用显存: {hw.effective_vram_gb:.1f} GB\n"
        md += f"- 模型目录可用: {hw.disk_free_gb:.1f} GB\n\n"
        if fallback:
            md += "⚠️ **注意**：以下模型不完全满足硬件要求，运行可能不稳定。\n\n"
        md += "### 推荐模型\n"
        for idx, m in enumerate(ranked, 1):
            md += f"**{idx}. {m['display_name']}** (`{m['ollama_tag']}`)\n"
            md += f"- 显存需求: {m['min_vram_gb']} GB | 内存: {m['min_ram_gb']} GB\n"
            md += f"- 编程能力: {m['capability']['coding']:.2f} | 中文能力: {m['capability']['chinese']:.2f}\n\n"
        return md

    def install_ui(model_tag: str, run_benchmark: bool):
        async def _install():
            if not await ensure_ollama_installed():
                return "❌ Ollama 未安装且自动安装失败，请手动安装。"
            await ensure_ollama_running()
            installed = await get_ollama_models()
            if model_tag in installed:
                msg = f"✅ 模型 `{model_tag}` 已安装，可直接运行"
                if run_benchmark and load_config().get("auto_benchmark", True):
                    tps = await benchmark_and_cache(model_tag)
                    msg += f"\n📊 实测速度: {tps:.1f} token/s"
                return msg
            await pull_model_with_retry(model_tag)
            msg = f"✅ 模型 `{model_tag}` 下载完成！\n可在终端使用 `ollama run {model_tag}` 启动。"
            if run_benchmark and load_config().get("auto_benchmark", True):
                tps = await benchmark_and_cache(model_tag)
                msg += f"\n📊 实测速度: {tps:.1f} token/s"
            return msg
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_install())
        finally:
            loop.close()

    with gr.Blocks(title="本地大模型助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 本地大模型智能推荐与安装")
        with gr.Row():
            with gr.Column(scale=1):
                purpose_input = gr.Textbox(label="你的用途", value="编程", placeholder="例如：编程、对话、写作")
                recommend_btn = gr.Button("🔍 推荐模型")
                output_recommend = gr.Markdown("点击按钮查看推荐")
            with gr.Column(scale=1):
                model_tag_input = gr.Textbox(label="Ollama 模型标签", placeholder="例如 qwen2.5:7b")
                benchmark_checkbox = gr.Checkbox(label="安装后运行性能测试", value=True)
                install_btn = gr.Button("📥 安装此模型")
                install_status = gr.Textbox(label="安装状态", interactive=False)
        recommend_btn.click(recommend_ui, inputs=purpose_input, outputs=output_recommend)
        install_btn.click(install_ui, inputs=[model_tag_input, benchmark_checkbox], outputs=install_status)

    demo.launch(server_name="127.0.0.1", server_port=7860)

# =========================
# 桌面 GUI (PyQt6)
# =========================
def serve_desktop_gui():
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QComboBox, QLineEdit, QPushButton, QListWidget,
            QTextEdit, QProgressBar, QMessageBox, QGroupBox
        )
        from PyQt6.QtCore import QThread, pyqtSignal, QTimer
    except ImportError:
        logger.error("请先安装 PyQt6: pip install PyQt6")
        sys.exit(1)

    hw = detect_hardware()
    models = load_models()

    class AsyncTask(QThread):
        output = pyqtSignal(str)
        finished = pyqtSignal()

        def __init__(self, coro_func, *args):
            super().__init__()
            self.coro_func = coro_func
            self.args = args

        def run(self):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.coro_func(*self.args))
                if result is not None:
                    self.output.emit(str(result))
            except Exception as e:
                self.output.emit(f"错误: {str(e)}")
            finally:
                self.finished.emit()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("本地大模型助手 (Ollama & LM Studio)")
            self.backend_mgr = BackendManager("ollama")
            self.init_ui()
            self.refresh_backend_status()

        def init_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout()

            # 后端选择区域
            backend_group = QGroupBox("后端选择")
            backend_layout = QHBoxLayout()
            backend_layout.addWidget(QLabel("推理后端:"))
            self.backend_combo = QComboBox()
            self.backend_combo.addItems(["Ollama", "LM Studio"])
            self.backend_combo.currentTextChanged.connect(self.on_backend_change)
            backend_layout.addWidget(self.backend_combo)
            self.backend_status_label = QLabel("状态: 检测中...")
            backend_layout.addWidget(self.backend_status_label)
            backend_group.setLayout(backend_layout)
            layout.addWidget(backend_group)

            # 用途推荐区域
            recommend_group = QGroupBox("模型推荐")
            recommend_layout = QHBoxLayout()
            recommend_layout.addWidget(QLabel("用途:"))
            self.purpose_input = QLineEdit("编程")
            recommend_layout.addWidget(self.purpose_input)
            self.recommend_btn = QPushButton("推荐模型")
            self.recommend_btn.clicked.connect(self.on_recommend)
            recommend_layout.addWidget(self.recommend_btn)
            recommend_group.setLayout(recommend_layout)
            layout.addWidget(recommend_group)

            # 模型列表
            self.model_list = QListWidget()
            layout.addWidget(self.model_list)

            # 安装/运行按钮
            self.install_btn = QPushButton("安装/运行所选模型")
            self.install_btn.clicked.connect(self.on_install)
            layout.addWidget(self.install_btn)

            # 日志区域
            log_group = QGroupBox("日志")
            log_layout = QVBoxLayout()
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            log_layout.addWidget(self.log_text)
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            log_layout.addWidget(self.progress_bar)
            log_group.setLayout(log_layout)
            layout.addWidget(log_group)

            central.setLayout(layout)

        def log(self, msg: str):
            self.log_text.append(msg)

        def on_backend_change(self, text: str):
            backend = "ollama" if text == "Ollama" else "lmstudio"
            self.backend_mgr = BackendManager(backend)
            self.refresh_backend_status()

        def refresh_backend_status(self):
            self.task = AsyncTask(self.backend_mgr.is_available)
            self.task.output.connect(self.update_backend_status)
            self.task.start()

        def update_backend_status(self, status_str: str):
            available = status_str == "True"
            self.backend_status_label.setText(f"状态: {'可用' if available else '不可用'}")
            if not available:
                self.log(f"{'Ollama' if self.backend_mgr.backend == 'ollama' else 'LM Studio'} 未运行")

        def on_recommend(self):
            purpose = self.purpose_input.text()
            if not purpose:
                QMessageBox.warning(self, "输入", "请输入用途")
                return
            self.task = AsyncTask(self.get_recommendations, purpose)
            self.task.output.connect(self.display_recommendations)
            self.task.start()

        async def get_recommendations(self, purpose: str) -> str:
            # 根据后端选择模型池
            if self.backend_mgr.backend == "ollama":
                model_pool = models  # models.json
            else:
                # LM Studio 从 API 获取
                model_pool = await get_lm_studio_models()
            ranked, fallback = recommend_models(hw, model_pool, purpose, top_k=5)
            result = json.dumps({"ranked": ranked, "fallback": fallback})
            return result

        def display_recommendations(self, result_json: str):
            try:
                data = json.loads(result_json)
                ranked = data["ranked"]
                fallback = data["fallback"]
                self.model_list.clear()
                for m in ranked:
                    tag = m.get("ollama_tag", m.get("display_name"))
                    self.model_list.addItem(f"{m['display_name']}  ({tag})")
                if fallback:
                    self.log("⚠️ 推荐结果不完全匹配硬件")
            except Exception as e:
                self.log(f"显示推荐出错: {e}")

        def on_install(self):
            selected = self.model_list.currentRow()
            if selected < 0:
                QMessageBox.warning(self, "选择模型", "请先在列表中选择一个模型")
                return
            # 重新获取推荐列表以取得完整模型信息（简单做法：再次获取）
            purpose = self.purpose_input.text()
            self.task = AsyncTask(self.install_model, purpose, selected)
            self.task.output.connect(self.log)
            self.task.finished.connect(lambda: self.progress_bar.setVisible(False))
            self.progress_bar.setVisible(True)
            self.task.start()

        async def install_model(self, purpose: str, index: int) -> str:
            try:
                if self.backend_mgr.backend == "ollama":
                    model_pool = models
                else:
                    model_pool = await get_lm_studio_models()
                ranked, _ = recommend_models(hw, model_pool, purpose, top_k=5)
                if index >= len(ranked):
                    return "无效的模型索引"
                model = ranked[index]
                tag = model.get("ollama_tag", model.get("display_name"))
                await self.backend_mgr.ensure_running()
                installed = await self.backend_mgr.get_installed_models()
                installed_tags = [m.get("ollama_tag", m.get("display_name")) for m in installed]
                if tag in installed_tags:
                    msg = f"模型 {tag} 已安装，正在启动..."
                    self.log(msg)
                    # 运行模型需要在另一个线程中避免阻塞 UI
                    QTimer.singleShot(500, lambda: self.run_model_in_thread(tag))
                    return msg
                else:
                    if self.backend_mgr.backend == "ollama":
                        await self.backend_mgr.pull_model(tag)
                        msg = f"模型 {tag} 下载完成，正在启动..."
                        self.log(msg)
                        QTimer.singleShot(500, lambda: self.run_model_in_thread(tag))
                        return msg
                    else:
                        return "LM Studio 模型请通过 LM Studio 界面下载后重试"
            except Exception as e:
                return f"安装失败: {e}"

        def run_model_in_thread(self, tag: str):
            # 在新线程中运行模型交互（会阻塞 GUI，但 Ollama/LM Studio 交互一般需要终端）
            # 这里简单启动一个子进程用于 ollama run
            if self.backend_mgr.backend == "ollama":
                subprocess.Popen(["ollama", "run", tag])
                self.log(f"已在终端启动 ollama run {tag}")
            else:
                # LM Studio 聊天暂时在终端运行，不做 GUI 内聊天
                self.log(f"请手动使用 LM Studio 与 {tag} 交互")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())

# =========================
# CLI 主流程 (集成后端选择)
# =========================
async def cli_main(purpose: str, list_only: bool, force_model: Optional[str],
                   prefer_speed: bool, skip_benchmark: bool, backend: str = "ollama"):
    logger.info("检测硬件...")
    hw = detect_hardware()
    print_hardware(hw)

    backend_mgr = BackendManager(backend)
    if not await backend_mgr.is_available():
        logger.warning(f"后端 {backend} 未运行，尝试启动...")
        try:
            await backend_mgr.ensure_running()
        except Exception as e:
            logger.error(str(e))
            return

    # 根据后端获取模型池
    if backend == "ollama":
        logger.info("加载 Ollama 模型数据库...")
        model_pool = load_models()
    else:
        logger.info("从 LM Studio 获取模型列表...")
        model_pool = await get_lm_studio_models()
        if not model_pool:
            logger.error("LM Studio 未提供任何模型")
            return

    if list_only:
        print_all_models(hw, model_pool)
        return

    if force_model:
        selected = None
        for m in model_pool:
            if m.get("ollama_tag") == force_model or m.get("display_name") == force_model:
                selected = m
                break
        if not selected:
            logger.error(f"模型 '{force_model}' 不在可用列表中")
            return
        ranked = [selected]
        fallback = False
    else:
        ranked, fallback = recommend_models(hw, model_pool, purpose, top_k=3, prefer_speed=prefer_speed)

    print_recommendations(ranked, fallback, hw.is_cpu_only)

    if len(ranked) > 1:
        choice = input(f"请选择要安装的模型编号 (1-{len(ranked)})，直接回车使用第一个: ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(ranked):
                    selected_model = ranked[idx]
                else:
                    logger.warning("无效编号，使用第一个模型")
                    selected_model = ranked[0]
            except ValueError:
                selected_model = ranked[0]
        else:
            selected_model = ranked[0]
    else:
        selected_model = ranked[0]
        confirm = input(f"\n是否安装并运行 {selected_model['display_name']}？[Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            logger.info("已取消")
            return

    tag = selected_model.get("ollama_tag") or selected_model["display_name"]

    # 对于 LM Studio，模型已经是安装好的，直接运行
    if backend == "lmstudio":
        logger.info(f"直接启动与 {tag} 的对话...")
        await backend_mgr.run_model(tag)
        return

    # Ollama 流程
    if not await ensure_ollama_installed():
        logger.error("Ollama 安装失败，请手动安装后重试")
        return

    await ensure_ollama_running()
    installed_tags = await get_ollama_models()
    if tag not in installed_tags:
        await pull_model_with_retry(tag)
    else:
        logger.info("模型已安装，跳过下载")

    if not skip_benchmark and load_config().get("auto_benchmark", True):
        real_tps = await benchmark_and_cache(tag)
        logger.info(f"实测速度: {real_tps:.1f} token/s")

    await run_ollama_interactive(tag)

# =========================
# 入口
# =========================
def main():
    parser = argparse.ArgumentParser(description="本地大模型推荐与安装工具（终极增强版）")
    parser.add_argument("--purpose", type=str, default="通用", help="用途：编程/对话/写作/翻译/通用")
    parser.add_argument("--list", action="store_true", help="列出所有模型及兼容性")
    parser.add_argument("--model", type=str, help="强制指定模型标签（跳过推荐）")
    parser.add_argument("--prefer-speed", action="store_true", help="偏好速度（降低模型大小）")
    parser.add_argument("--web", action="store_true", help="启动 Web UI 界面")
    parser.add_argument("--gui", action="store_true", help="启动桌面 GUI (PyQt6)")
    parser.add_argument("--update-models", action="store_true", help="从远程更新模型数据库")
    parser.add_argument("--skip-benchmark", action="store_true", help="跳过安装后的性能测试")
    parser.add_argument("--backend", type=str, choices=["ollama", "lmstudio"], default="ollama",
                        help="选择推理后端 (ollama 或 lmstudio，CLI 模式有效)")
    args = parser.parse_args()

    if args.update_models:
        if update_models_from_remote():
            print("模型数据库更新完成")
        else:
            print("更新失败，请检查网络或 config.json 中的 remote_url")
        return

    if args.web:
        serve_web_ui()
    elif args.gui:
        serve_desktop_gui()
    else:
        try:
            asyncio.run(cli_main(
                purpose=args.purpose,
                list_only=args.list,
                force_model=args.model,
                prefer_speed=args.prefer_speed,
                skip_benchmark=args.skip_benchmark,
                backend=args.backend
            ))
        except KeyboardInterrupt:
            logger.info("用户中断")
            sys.exit(0)
        except Exception as e:
            logger.exception(e)
            sys.exit(1)

if __name__ == "__main__":
    main()
