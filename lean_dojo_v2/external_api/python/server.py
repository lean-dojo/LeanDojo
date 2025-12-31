import gc
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from external_models import *
from fastapi import FastAPI, HTTPException
from leanprogress import LeanProgressScorer
from models import *
from pydantic import BaseModel

app = FastAPI()


@dataclass
class ModelBundle:
    generator: Generator
    progress_scorer: Optional[LeanProgressScorer] = None


def _build_progress_scorer() -> Optional[LeanProgressScorer]:
    model_path = os.getenv("LEANPROGRESS_MODEL", "").strip()
    if not model_path:
        return None
    device = os.getenv("LEANPROGRESS_DEVICE", "auto").strip()
    template = os.getenv("LEANPROGRESS_TEMPLATE", None)
    try:
        return LeanProgressScorer(
            model_name=model_path,
            device=device,
            template=template,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LeanProgress model '{model_path}': {exc}"
        ) from exc


models = {
    "deepseek": ModelBundle(
        generator=HFTacticGenerator(
            model="deepseek-ai/DeepSeek-Prover-V2-671B:novita",
            temperature=0.6,
            max_new_tokens=256,
            top_p=0.9,
            length_penalty=0,
            num_return_sequences=4,
            do_sample=True,
            output_scores=True,
            output_logits=False,
            return_dict_in_generate=True,
            device="auto",
        ),
        progress_scorer=_build_progress_scorer(),
    ),
}


class GeneratorRequest(BaseModel):
    name: str
    input: str
    prefix: Optional[str]
    use_reward: bool = False


class Generation(BaseModel):
    output: str
    score: float
    model_score: Optional[float] = None
    steps_remaining: Optional[float] = None
    reward: Optional[float] = None


class GeneratorResponse(BaseModel):
    outputs: List[Generation]


class EncoderRequest(BaseModel):
    name: str
    input: str


class EncoderResponse(BaseModel):
    outputs: List[float]


@app.post("/generate")
async def generate(req: GeneratorRequest) -> GeneratorResponse:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    try:
        bundle = models[req.name]
        target_prefix = req.prefix if req.prefix is not None else ""
        outputs = bundle.generator.generate(req.input, target_prefix)
        generations: List[Generation] = []
        if req.use_reward:
            if bundle.progress_scorer is None:
                raise HTTPException(
                    status_code=400,
                    detail="LeanProgress scoring requested but no model configured.",
                )
            for out, model_score in outputs:
                steps_remaining = bundle.progress_scorer.predict(
                    goal=req.input,
                    tactic=out,
                    prefix=req.prefix,
                )
                reward_value = -steps_remaining
                generations.append(
                    Generation(
                        output=out,
                        score=reward_value,
                        model_score=model_score,
                        steps_remaining=steps_remaining,
                        reward=reward_value,
                    )
                )
        else:
            generations = [
                Generation(output=out, score=score) for out, score in outputs
            ]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return GeneratorResponse(outputs=generations)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(status_code=500, detail="GPU out of memory")
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode")
async def encode(req: EncoderRequest) -> EncoderResponse:
    bundle = models[req.name]
    if not hasattr(bundle.generator, "encode"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.name}' does not support encoding requests.",
        )
    feature = bundle.generator.encode(req.input)  # type: ignore[attr-defined]
    return EncoderResponse(outputs=feature.tolist())
