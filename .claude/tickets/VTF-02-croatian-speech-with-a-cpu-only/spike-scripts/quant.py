"""Quantize the bidirectional LLM export (scratch venv). usage: quant.py <in.onnx> <out_dir> <mode: int8|int4>"""
import sys, os
import onnx

src, out_dir, mode = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
if mode == "int8":
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = os.path.join(out_dir, "llm_bidir_int8.onnx")
    quantize_dynamic(src, out, weight_type=QuantType.QInt8, per_channel=True, use_external_data_format=False)
else:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    out = os.path.join(out_dir, "llm_bidir_int4.onnx")
    m = onnx.load(src)
    q = MatMulNBitsQuantizer(m, block_size=128, is_symmetric=True, bits=4)
    q.process()
    q.model.save_model_to_file(out, True)
print("saved", out, os.path.getsize(out))
