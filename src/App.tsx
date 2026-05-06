import React, { useRef, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Eraser, 
  Upload, 
  Send, 
  RefreshCcw, 
  BrainCircuit, 
  Lightbulb,
  Maximize2,
  AlertCircle
} from 'lucide-react';

// Design Constants
const CANVAS_SIZE = 280;
const MODEL_SIZE = 28;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'draw' | 'upload'>('draw');

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.lineWidth = 20;
        ctx.strokeStyle = 'white';
      }
    }
  }, [mode]);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.beginPath(); // Start a new path for next stroke
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = ('touches' in e) ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = ('touches' in e) ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
    setPrediction(null);
    setConfidence(null);
    setError(null);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (ctx) {
          ctx.fillStyle = 'black';
          ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
          // Scale to fit
          const scale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height);
          const x = (CANVAS_SIZE - img.width * scale) / 2;
          const y = (CANVAS_SIZE - img.height * scale) / 2;
          ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
        }
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const getCanvasData = () => {
    const canvas = canvasRef.current;
    if (!canvas) return [];

    // Create a temporary 28x28 canvas to downsample
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MODEL_SIZE;
    tempCanvas.height = MODEL_SIZE;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return [];

    tempCtx.drawImage(canvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE);
    const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const pixels = [];

    // Convert to grayscale and normalize (0 to 1)
    for (let i = 0; i < imageData.data.length; i += 4) {
      // MNIST is usually white digits on black background
      // Pixel intensity is the R, G, or B value (they are the same for gray)
      // Standard MNIST training uses center and grayscale
      const avg = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
      pixels.push(avg / 255.0);
    }

    return pixels;
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    const pixels = getCanvasData();

    try {
      // In a real scenario, this matches the FastAPI server URL
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: pixels }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}. Make sure the FastAPI backend is running.`);
      }

      const result = await response.json();
      setPrediction(result.prediction);
      setConfidence(result.confidence);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to connect to the backend.');
      
      // MOCK for preview purposes if backend is missing
      if (err.message.includes('Failed to fetch') || err.message.includes('404')) {
        setTimeout(() => {
          setPrediction("?");
          setConfidence(0);
          setError("Development Note: The FastAPI backend is not running. Please start it using 'uvicorn app:app'.");
        }, 800);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-white selection:bg-indigo-500/30 flex items-center justify-center p-4 font-sans">
      <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
        
        {/* Left Side: Header & Controls */}
        <div className="md:col-span-5 space-y-6">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium uppercase tracking-wider">
              <BrainCircuit className="w-3 h-3" />
              Machine Learning
            </div>
            <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-br from-white to-white/60 bg-clip-text text-transparent">
              MNIST Digit<br />Recognizer
            </h1>
            <p className="text-gray-400 text-sm leading-relaxed max-w-sm">
              Draw a digit from 0 to 9 on the canvas or upload an image. 
              Our neural network will predict what you've written.
            </p>
          </motion.div>

          <div className="space-y-4">
            <div className="flex gap-2 p-1 bg-white/5 rounded-lg border border-white/10">
              <button 
                onClick={() => setMode('draw')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === 'draw' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' : 'text-gray-400 hover:text-white hover:bg-white/5'}`}
              >
                <Eraser className="w-4 h-4" />
                Draw
              </button>
              <button 
                onClick={() => setMode('upload')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-all ${mode === 'upload' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' : 'text-gray-400 hover:text-white hover:bg-white/5'}`}
              >
                <Upload className="w-4 h-4" />
                Upload
              </button>
            </div>

            <div className="p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                Pro-tip
              </h3>
              <p className="text-xs text-gray-500 leading-normal">
                For best results, draw centrally and boldly. The model was trained on clear, centered handwritten digits.
              </p>
            </div>
          </div>
        </div>

        {/* Right Side: Canvas & Results */}
        <div className="md:col-span-7 space-y-6">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative"
          >
            {/* Main Interactive Box */}
            <div className="bg-[#111114] border border-white/10 rounded-2xl p-6 shadow-2xl relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent pointer-events-none" />
              
              <div className="relative z-10 space-y-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-indigo-400 uppercase tracking-widest px-2 py-0.5 border border-indigo-500/30 rounded bg-indigo-500/10">
                    Input Stage
                  </span>
                  <div className="flex gap-2">
                    <button 
                      onClick={clearCanvas}
                      className="p-2 rounded-lg bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                      title="Clear Canvas"
                    >
                      <RefreshCcw className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="flex justify-center">
                  <div className="relative p-1 bg-white/10 rounded-xl overflow-hidden shadow-inner ring-1 ring-white/20">
                    <canvas
                      ref={canvasRef}
                      width={CANVAS_SIZE}
                      height={CANVAS_SIZE}
                      onMouseDown={startDrawing}
                      onMouseMove={draw}
                      onMouseUp={stopDrawing}
                      onMouseLeave={stopDrawing}
                      onTouchStart={startDrawing}
                      onTouchMove={draw}
                      onTouchEnd={stopDrawing}
                      className="rounded-lg cursor-crosshair touch-none bg-black"
                    />
                    {mode === 'upload' && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
                         <div className="text-center p-4">
                           <Upload className="w-8 h-8 text-indigo-400 mx-auto mb-2" />
                           <p className="text-xs font-medium">Select Image File</p>
                         </div>
                      </div>
                    )}
                    {mode === 'upload' && (
                      <input 
                        type="file" 
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="absolute inset-0 opacity-0 cursor-pointer"
                      />
                    )}
                  </div>
                </div>

                <button
                  disabled={loading}
                  onClick={handlePredict}
                  className="w-full flex items-center justify-center gap-2 py-3 px-6 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-lg shadow-indigo-600/20 active:scale-[0.98]"
                >
                  {loading ? (
                    <RefreshCcw className="w-5 h-5 animate-spin" />
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      Classify Image
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Results Display */}
            <AnimatePresence>
              {(prediction !== null || error) && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="mt-6 space-y-4"
                >
                  {error && (
                    <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex gap-3 text-red-400">
                      <AlertCircle className="w-5 h-5 shrink-0" />
                      <p className="text-xs uppercase font-medium tracking-tight">{error}</p>
                    </div>
                  )}

                  {prediction !== null && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#111114] border border-white/10 rounded-xl p-4 text-center">
                        <span className="text-[10px] font-mono text-gray-500 uppercase block mb-1">Prediction</span>
                        <div className="text-5xl font-bold text-indigo-400">{prediction}</div>
                      </div>
                      <div className="bg-[#111114] border border-white/10 rounded-xl p-4 text-center">
                        <span className="text-[10px] font-mono text-gray-500 uppercase block mb-1">Confidence</span>
                        <div className="text-5xl font-bold text-indigo-400">
                          {confidence !== null ? `${Math.round(confidence * 100)}%` : '??'}
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 text-[10px] text-gray-600 font-mono flex items-center gap-4 whitespace-nowrap">
        <span>MODEL: Scikit-learn MLP</span>
        <span className="w-1 h-1 bg-gray-800 rounded-full" />
        <span>BACKEND: FastAPI</span>
        <span className="w-1 h-1 bg-gray-800 rounded-full" />
        <span>UI: React 19 + Framer Motion</span>
      </div>
    </div>
  );
}

