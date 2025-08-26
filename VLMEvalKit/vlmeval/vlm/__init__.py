import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)

try:
    from .aria import Aria
except ImportError:
    pass

from .base import BaseModel

try:
    from .hawk_vl import HawkVL
except ImportError:
    pass

try:
    from .cogvlm import CogVlm, GLM4v
except ImportError:
    pass

try:
    from .emu import Emu, Emu3_chat, Emu3_gen
except ImportError:
    pass

try:
    from .eagle_x import Eagle
except ImportError:
    pass

try:
    from .granite_vision import GraniteVision3
except ImportError:
    pass

try:
    from .idefics import IDEFICS, IDEFICS2
except ImportError:
    pass

try:
    from .instructblip import InstructBLIP
except ImportError:
    pass

try:
    from .kosmos import Kosmos2
except ImportError:
    pass

try:
    from .llava import (
        LLaVA,
        LLaVA_Next,
        LLaVA_XTuner,
        LLaVA_Next2,
        LLaVA_OneVision,
        LLaVA_OneVision_HF,
    )
except ImportError:
    pass

try:
    from .vita import VITA, VITAQwen2
except ImportError:
    pass

try:
    from .long_vita import LongVITA
except ImportError:
    pass

try:
    from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6, MiniCPM_o_2_6
except ImportError:
    pass

try:
    from .minigpt4 import MiniGPT4
except ImportError:
    pass

try:
    from .mmalaya import MMAlaya, MMAlaya2
except ImportError:
    pass

try:
    from .monkey import Monkey, MonkeyChat
except ImportError:
    pass

try:
    from .moondream import Moondream1, Moondream2
except ImportError:
    pass

try:
    from .minimonkey import MiniMonkey
except ImportError:
    pass

try:
    from .mplug_owl2 import mPLUG_Owl2
except ImportError:
    pass

try:
    from .omnilmm import OmniLMM12B
except ImportError:
    pass

try:
    from .open_flamingo import OpenFlamingo
except ImportError:
    pass

try:
    from .pandagpt import PandaGPT
except ImportError:
    pass

try:
    from .qwen_vl import QwenVL, QwenVLChat
except ImportError:
    pass

try:
    from .qwen2_vl import Qwen2VLChat, Qwen2VLChatAguvis
except ImportError:
    pass

try:
    from .transcore_m import TransCoreM
except ImportError:
    pass

try:
    from .visualglm import VisualGLM
except ImportError:
    pass

try:
    from .xcomposer import (
        ShareCaptioner,
        XComposer,
        XComposer2,
        XComposer2_4KHD,
        XComposer2d5,
    )
except ImportError:
    pass

try:
    from .yi_vl import Yi_VL
except ImportError:
    pass

try:
    from .internvl import InternVLChat
except ImportError:
    pass

try:
    from .deepseek_vl import DeepSeekVL
except ImportError:
    pass

try:
    from .deepseek_vl2 import DeepSeekVL2
except ImportError:
    pass

try:
    from .janus import Janus
except ImportError:
    pass

try:
    from .mgm import Mini_Gemini
except ImportError:
    pass

try:
    from .bunnyllama3 import BunnyLLama3
except ImportError:
    pass

try:
    from .vxverse import VXVERSE
except ImportError:
    pass

try:
    from .gemma import PaliGemma, Gemma3
except ImportError:
    pass

try:
    from .qh_360vl import QH_360VL
except ImportError:
    pass

try:
    from .phi3_vision import Phi3Vision, Phi3_5Vision
except ImportError:
    pass

try:
    from .phi4_multimodal import Phi4Multimodal
except ImportError:
    pass

try:
    from .wemm import WeMM
except ImportError:
    pass

try:
    from .cambrian import Cambrian
except ImportError:
    pass

try:
    from .chameleon import Chameleon
except ImportError:
    pass

try:
    from .video_llm import (
        VideoLLaVA,
        VideoLLaVA_HF,
        Chatunivi,
        VideoChatGPT,
        LLaMAVID,
        VideoChat2_HD,
        PLLaVA,
    )
except ImportError:
    pass

try:
    from .vila import VILA, NVILA
except ImportError:
    pass

try:
    from .ovis import Ovis, Ovis1_6, Ovis1_6_Plus, Ovis2, OvisU1
except ImportError:
    pass

try:
    from .mantis import Mantis
except ImportError:
    pass

try:
    from .mixsense import LLama3Mixsense
except ImportError:
    pass

try:
    from .parrot import Parrot
except ImportError:
    pass

try:
    from .omchat import OmChat
except ImportError:
    pass

try:
    from .rbdash import RBDash
except ImportError:
    pass

try:
    from .xgen_mm import XGenMM
except ImportError:
    pass

try:
    from .slime import SliME
except ImportError:
    pass

try:
    from .mplug_owl3 import mPLUG_Owl3
except ImportError:
    pass

try:
    from .pixtral import Pixtral
except ImportError:
    pass

try:
    from .llama_vision import llama_vision
except ImportError:
    pass

try:
    from .llama4 import llama4
except ImportError:
    pass

try:
    from .molmo import molmo
except ImportError:
    pass

try:
    from .points import POINTS, POINTSV15
except ImportError:
    pass

try:
    from .nvlm import NVLM
except ImportError:
    pass

try:
    from .vintern_chat import VinternChat
except ImportError:
    pass

try:
    from .h2ovl_mississippi import H2OVLChat
except ImportError:
    pass

try:
    from .falcon_vlm import Falcon2VLM
except ImportError:
    pass

try:
    from .smolvlm import SmolVLM, SmolVLM2
except ImportError:
    pass

try:
    from .sail_vl import SailVL
except ImportError:
    pass

try:
    from .valley import Valley2Chat
except ImportError:
    pass

try:
    from .ross import Ross
except ImportError:
    pass

try:
    from .ola import Ola
except ImportError:
    pass

try:
    from .x_vl import X_VL_HF
except ImportError:
    pass

try:
    from .ursa import UrsaChat
except ImportError:
    pass

try:
    from .vlm_r1 import VLMR1Chat
except ImportError:
    pass

try:
    from .aki import AKI
except ImportError:
    pass

try:
    from .ristretto import Ristretto
except ImportError:
    pass

try:
    from .vlaa_thinker import VLAAThinkerChat
except ImportError:
    pass

try:
    from .kimi_vl import KimiVL
except ImportError:
    pass

try:
    from .wethink_vl import WeThinkVL
except ImportError:
    pass

try:
    from .flash_vl import FlashVL
except ImportError:
    pass

try:
    from .oryx import Oryx
except ImportError:
    pass

try:
    from .treevgr import TreeVGR
except ImportError:
    pass

try:
    from .glm4_1v import GLM4_1v
except ImportError:
    pass

try:
    from .varco_vision import VarcoVision
except ImportError:
    pass

try:
    from .qtunevl import QTuneVL, QTuneVLChat
except ImportError:
    pass

from .socrates_mitm import ZanobiaMitM, from_name
