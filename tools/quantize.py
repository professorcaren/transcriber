"""Dynamic int8 quantization of models/step.onnx, plus the mel filterbank used by diar.js."""
import librosa
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic("models/step.onnx", "models/step_int8.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney").astype("<f4").tofile("models/mel_filters.bin")
