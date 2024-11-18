'use client'

import { useState, useRef, useEffect } from 'react'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [autoMode, setAutoMode] = useState(true)
  const [points, setPoints] = useState<[number, number][]>([])
  const [showCanvas, setShowCanvas] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [result, setResult] = useState<{
    success: boolean
    total_fringes?: number
    error?: string
  } | null>(null)
  const [progress, setProgress] = useState<{
    total_frames: number
    current_frame: number
    is_processing: boolean
  }>({
    total_frames: 0,
    current_frame: 0,
    is_processing: false
  })

  // 轮询处理进度
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    if (isProcessing) {
      intervalId = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/processing-progress')
          const data = await response.json()
          setProgress(data)
        } catch (error) {
          console.error('获取进度时发生错误:', error)
        }
      }, 500)
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
    }
  }, [isProcessing])

  const resetStates = () => {
    setFile(null)
    setPoints([])
    setShowCanvas(false)
    setResult(null)
    setProgress({
      total_frames: 0,
      current_frame: 0,
      is_processing: false
    })
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setResult(null)
      
      if (!autoMode) {
        // 如果是手动模式，显示第一帧用于选点
        const reader = new FileReader()
        reader.onload = (e) => {
          const img = new Image()
          img.onload = () => {
            if (canvasRef.current) {
              const canvas = canvasRef.current
              canvas.width = img.width
              canvas.height = img.height
              const ctx = canvas.getContext('2d')
              ctx?.drawImage(img, 0, 0)
            }
          }
          img.src = e.target?.result as string
        }
        reader.readAsDataURL(selectedFile)
        setShowCanvas(true)
        setPoints([])
      }
    }
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (points.length >= 2) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const newPoint: [number, number] = [x, y]
    
    setPoints(prev => {
      const newPoints = [...prev, newPoint]
      
      // 绘制点和线
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.fillStyle = '#6366f1' // Indigo point color
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
        
        if (newPoints.length === 2) {
          ctx.strokeStyle = '#8b5cf6' // Purple line color
          ctx.beginPath()
          ctx.moveTo(newPoints[0][0], newPoints[0][1])
          ctx.lineTo(newPoints[1][0], newPoints[1][1])
          ctx.stroke()
        }
      }
      
      return newPoints
    })
  }

  const handleSubmit = async () => {
    if (!file) return

    setIsProcessing(true)
    setResult(null)
    setProgress({
      total_frames: 0,
      current_frame: 0,
      is_processing: true
    })

    const formData = new FormData()
    formData.append('file', file)
    formData.append('auto_mode', autoMode.toString())
    if (!autoMode && points.length === 2) {
      formData.append('points', JSON.stringify(points))
    }

    try {
      const response = await fetch('http://localhost:8000/process-video', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      setResult({
        success: false,
        error: '处理过程中发生错误',
      })
    } finally {
      setIsProcessing(false)
      setProgress(prev => ({ ...prev, is_processing: false }))
    }
  }

  const handleDownload = async () => {
    try {
      const response = await fetch('http://localhost:8000/download-video')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'processed_video.mp4'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('下载视频时发生错误:', error)
    }
  }

  const resetPoints = () => {
    setPoints([])
    if (canvasRef.current && file) {
      // 重新加载图片
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          if (canvasRef.current) {
            const canvas = canvasRef.current
            const ctx = canvas.getContext('2d')
            ctx?.clearRect(0, 0, canvas.width, canvas.height)
            ctx?.drawImage(img, 0, 0)
          }
        }
        img.src = e.target?.result as string
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-indigo-100 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-indigo-900 mb-4">
            干涉条纹追踪器
          </h1>
          <p className="text-lg text-indigo-600">
            上传视频文件，自动计算干涉条纹的移动数量
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-md p-8 mb-8 border border-indigo-100">
          {/* 处理模式选择 */}
          <div className="mb-6">
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setAutoMode(true)}
                className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                  autoMode
                    ? 'bg-indigo-600 text-white shadow-md'
                    : 'bg-indigo-100 text-indigo-600'
                }`}
              >
                自动模式
              </button>
              <button
                onClick={() => setAutoMode(false)}
                className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                  !autoMode
                    ? 'bg-indigo-600 text-white shadow-md'
                    : 'bg-indigo-100 text-indigo-600'
                }`}
              >
                手动模式
              </button>
            </div>
            <p className="text-sm text-indigo-500 text-center mt-2">
              {autoMode
                ? '自动选择一条竖直线进行分析'
                : '手动选择两个点确定分析线'}
            </p>
          </div>

          {/* 文件上传区域 */}
          <div className="mb-8">
            <div className="flex items-center justify-center w-full">
              <label
                htmlFor="dropzone-file"
                className="flex flex-col items-center justify-center w-full h-64 border-2 border-indigo-200 border-dashed rounded-lg cursor-pointer bg-indigo-50 hover:bg-indigo-100 transition-colors duration-200"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <svg
                    className="w-10 h-10 mb-3 text-indigo-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    ></path>
                  </svg>
                  <p className="mb-2 text-sm text-indigo-600">
                    <span className="font-semibold">点击上传</span> 或拖放文件
                  </p>
                  <p className="text-xs text-indigo-500">支持MP4、MOV等常见视频格式</p>
                </div>
                <input
                  id="dropzone-file"
                  type="file"
                  className="hidden"
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </label>
            </div>
            {file && (
              <p className="mt-4 text-sm text-indigo-500 text-center">
                已选择文件: {file.name}
              </p>
            )}
          </div>

          {/* 手动选点画布 */}
          {showCanvas && (
            <div className="mb-8">
              <div className="relative">
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  className="mx-auto border border-indigo-200 rounded-lg"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
                <div className="absolute top-2 right-2">
                  <button
                    onClick={resetPoints}
                    className="px-3 py-1 bg-indigo-600 text-white rounded-lg text-sm shadow-md hover:bg-indigo-700 transition-colors duration-200"
                  >
                    重选
                  </button>
                </div>
              </div>
              <p className="text-sm text-indigo-500 text-center mt-2">
                {points.length === 2
                  ? '已选择两点，可以开始处理'
                  : `请在图像上选择${2 - points.length}个点`}
              </p>
            </div>
          )}

          {/* 提交按钮和进度条 */}
          <div className="flex flex-col items-center space-y-4">
            {isProcessing && progress.total_frames > 0 && (
              <div className="w-full">
                <div className="flex justify-between text-sm text-indigo-600 mb-1">
                  <span>处理进度</span>
                  <span>{Math.round((progress.current_frame / progress.total_frames) * 100)}%</span>
                </div>
                <div className="w-full bg-indigo-100 rounded-full h-2">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${(progress.current_frame / progress.total_frames) * 100}%`
                    }}
                  />
                </div>
              </div>
            )}
            
            <button
              onClick={handleSubmit}
              disabled={!file || isProcessing || (!autoMode && points.length !== 2)}
              className={`px-6 py-3 rounded-lg text-white font-medium shadow-md transition-all duration-200 ${
                !file || isProcessing || (!autoMode && points.length !== 2)
                  ? 'bg-indigo-300 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {isProcessing ? '处理中...' : '开始处理'}
            </button>
          </div>
        </div>

        {/* 结果显示 */}
        {result && (
          <div
            className={`bg-white rounded-xl shadow-md p-8 ${
              result.success ? 'border-purple-500' : 'border-red-500'
            } border-2`}
          >
            {result.success ? (
              <div className="text-center">
                <h2 className="text-2xl font-bold text-indigo-900 mb-4">
                  处理完成
                </h2>
                <p className="text-lg text-indigo-700 mb-6">
                  总移动条纹数: {result.total_fringes}
                </p>
                <button
                  onClick={handleDownload}
                  className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium shadow-md transition-colors duration-200"
                >
                  下载处理后的视频
                </button>
              </div>
            ) : (
              <div className="text-center">
                <h2 className="text-2xl font-bold text-red-600 mb-4">处理失败</h2>
                <p className="text-indigo-700">{result.error}</p>
              </div>
            )}
          </div>
        )}

        {/* 处理新文件按钮 */}
        {result && (
          <div className="mt-6 text-center">
            <button
              onClick={resetStates}
              className="px-6 py-3 bg-indigo-100 text-indigo-600 rounded-lg font-medium hover:bg-indigo-200 transition-colors duration-200"
            >
              处理新文件
            </button>
          </div>
        )}
      </div>
    </main>
  )
}
