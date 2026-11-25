"""Pin the vendor prepare implementation, suppress the fallback text and record the decoder's input IDs."""
import ast
import hashlib
import inspect
import json
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch

PINS={
 "minicpm-o-4.5-local":("MiniCPMODuplex","02b5a516408fd835c4591d0850d8c029e25a8821a322aaa1be7cb1969112f01e"),
 "realtime-venus-omni":("RealtimeVenusOmniDuplex","a700847af79f95ef3c87d974c5ac229210a01fe46eaf442e531f418f51450b80")}
ROLE_PREFIX="<|im_start|>system\n"
ROLE_SUFFIX="<|im_end|>"


def vendor_function(source,path,model,namespace):
    tree=ast.parse(source)
    name,expected=PINS[model]
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==name)
    node=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=="prepare")
    before=ast.get_source_segment(source,node)
    if hashlib.sha256(before.encode()).hexdigest()!=expected or node.decorator_list:
        raise ValueError("Unverified native duplex prepare code "+model)
    matches=[(i,s) for i,s in enumerate(node.body) if isinstance(s,ast.Assign) and isinstance(s.value,ast.BoolOp)
             and len(s.targets)==1 and isinstance(s.targets[0],ast.Name) and s.targets[0].id=="prefix_system_prompt"]
    if len(matches)!=1:raise ValueError("No pinned default assignment")
    i,stmt=matches[0]
    node.body[i]=ast.copy_location(ast.parse('prefix_system_prompt = ""').body[0],stmt)
    subset=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    dest={};exec(compile(subset,str(path),"exec"),namespace,dest)
    return dest["prepare"]


MEMORY_SHA="b14ccd6bd090abd95d4b72bd54a7bde644bcf680c12e81b633aec85fb29e5df8"
def memory_without_default(runtime):
    original=type(runtime).prepare
    path=Path(inspect.getsourcefile(original))
    source=path.read_text()
    matching=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=="prepare"
              and hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()==MEMORY_SHA]
    if len(matching)!=1:raise ValueError("Unknown Venus memory.prepare")
    node=matching[0]
    hit=[s for s in node.body if isinstance(s,ast.Assign) and ast.unparse(s)=="prompt = prefix_system_prompt or self.config.system_prompt"]
    if len(hit)!=1:raise ValueError("Unknown memory prompt fallback")
    hit[0].value=ast.Constant(value="")
    dest={}
    code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(code,str(path),"exec"),original.__globals__,dest)
    return MethodType(dest["prepare"],runtime)


def prepare_no_text(duplex,model,output):
    official=getattr(duplex,"official_duplex",duplex)
    original=type(official).prepare
    path=Path(inspect.getsourcefile(original))
    if getattr(official,"generate_audio",None) is not False:
        raise ValueError("Unexpected generation mode")
    native=vendor_function(path.read_text(),path,model,original.__globals__)
    token=official.tokenizer
    prefix,suffix=(token.encode(s,add_special_tokens=False) for s in (ROLE_PREFIX,ROLE_SUFFIX))
    expected=prefix+suffix
    used=[]
    calls=[]
    saved_embed=official.decoder.embed_token
    saved_encode=token.encode
    def guarded_encode(text,*args,**kwargs):
        if text not in ("",ROLE_PREFIX,ROLE_SUFFIX):
            raise ValueError("Unapproved natural-language or template input during native startup")
        ids=saved_encode(text,*args,**kwargs)
        if text=="" and ids:
            raise ValueError("The empty sliding-window marker produced input tokens")
        calls.append({"protocol_text":text,"ids":list(ids)})
        return ids
    def guarded_embed(token_id,*args,**kwargs):
        index=len(used);v=int(token_id)
        if index>=len(expected) or v!=expected[index]:
            raise ValueError("Unexpected embedding lookup in native system initialization")
        used.append(v);return saved_embed(token_id,*args,**kwargs)
    def invoke(self,*args,**kwargs):
        supplied=inspect.signature(native).bind(self,*args,**kwargs);supplied.apply_defaults()
        if supplied.arguments["prefix_system_prompt"] not in (None,"") or supplied.arguments["ref_audio"] is not None or supplied.arguments["prompt_wav_path"] is not None:
            raise ValueError("Unexpected initialization content from outer duplex wrapper")
        return native(self,prefix_system_prompt="",context_previous_marker="")
    memory=getattr(duplex,"memory_runtime",None)
    with ExitStack() as stack:
        if memory is not None:
            stack.enter_context(patch.object(memory,"prepare",memory_without_default(memory)))
        stack.enter_context(patch.object(official,"prepare",MethodType(invoke,official)))
        stack.enter_context(patch.object(token,"encode",guarded_encode))
        stack.enter_context(patch.object(official.decoder,"embed_token",guarded_embed))
        prepared=duplex.prepare(prefix_system_prompt="",context_previous_marker="")
    if used!=expected or prepared!=ROLE_PREFIX+ROLE_SUFFIX:
        raise ValueError(("Strict native input trace disagrees",used,expected,prepared))
    evidence=dict(schema="omni-life-tts-zero-text-boundary/v1",model=model,original_prepare_sha256=PINS[model][1],
                  memory_prepare_sha256=MEMORY_SHA if memory is not None else None,native_decoder_embedding_ids=used,expected_role_control_ids=expected,tokenizer_calls=calls,
                  nested_native_prepared_prompt=prepared,system_body="",context_previous_marker="",no_natural_language_input=True)
    with (Path(output)/"strict-input-token-receipt.json").open("x") as f:json.dump(evidence,f,ensure_ascii=False,indent=2)
    return prepared


def prefill_no_text(duplex,audio,frames,ledger,phase):
    official=getattr(duplex,"official_duplex",duplex)
    token=official.tokenizer
    saved=token.encode
    calls=[]
    def only_control(text,*args,**kwargs):
        if text not in ("",ROLE_PREFIX,ROLE_SUFFIX) and text not in token.all_special_tokens:
            raise ValueError("Unexpected text from wrapper/processor during AV prefill")
        ids=saved(text,*args,**kwargs)
        calls.append(dict(protocol_text=text,ids=list(ids)))
        return ids
    def dims(result):
        if hasattr(result,"shape"):return [list(result.shape)]
        if isinstance(result,(tuple,list)):return [shape for segment in result for shape in dims(segment)]
        return []
    names=("unit_token_id","image_start_token_id","image_end_token_id","slice_start_token_id","slice_end_token_id")
    allowed={int(getattr(official,n)):n for n in names}
    token_lookups=[];encoded={"audio":[],"vision":[]};fed=[]
    embed=official.decoder.embed_token
    feed=official.decoder.feed
    def only_structural(token_id,*args,**kwargs):
        if hasattr(token_id,"shape") or int(token_id) not in allowed:
            raise ValueError("Unexpected text-token embedding in audiovisual prefill")
        token_lookups.append({"id":int(token_id),"protocol_name":allowed[int(token_id)]})
        return embed(token_id,*args,**kwargs)
    def observed_feed(embedding,*args,**kwargs):
        fed.extend(dims(embedding));return feed(embedding,*args,**kwargs)
    def encoder_callback(original,channel):
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs);encoded[channel].extend(dims(result));return result
        return wrapped
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(token,"encode",only_control))
            stack.enter_context(patch.object(official.decoder,"embed_token",only_structural))
            stack.enter_context(patch.object(official.decoder,"feed",observed_feed))
            for name,channel in (("get_audio_embedding_streaming","audio"),("get_vision_embedding","vision")):
                stack.enter_context(patch.object(official.model,name,encoder_callback(getattr(official.model,name),channel)))
            return duplex.streaming_prefill(audio,frames,None,1,False)
    finally:
        ledger.event("input.jsonl","strict_audio_encoder_boundary",input_origin=phase,text_list=None,
                     audio_samples=0 if audio is None else len(audio),frames=len(frames),tokenizer_calls=calls,
                     input_token_lookups=token_lookups,input_decoder_feed_shapes=fed,encoded_embedding_shapes=encoded,
                     submitted_float32_sha256=hashlib.sha256(audio.tobytes()).hexdigest() if audio is not None else None,
                     submitted_rgb_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in frames])
