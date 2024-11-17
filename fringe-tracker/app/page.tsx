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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      
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
        ctx.fillStyle = 'red'
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
        
        if (newPoints.length === 2) {
          ctx.strokeStyle = 'green'
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
      setShowCanvas(false)
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
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            干涉条纹追踪器
          </h1>
          <p className="text-lg text-gray-600">
            上传视频文件，自动计算干涉条纹的移动数量
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          {/* 处理模式选择 */}
          <div className="mb-6">
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setAutoMode(true)}
                className={`px-4 py-2 rounded-lg ${
                  autoMode
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                自动模式
              </button>
              <button
                onClick={() => setAutoMode(false)}
                className={`px-4 py-2 rounded-lg ${
                  !autoMode
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                手动模式
              </button>
            </div>
            <p className="text-sm text-gray-500 text-center mt-2">
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
                className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <svg
                    className="w-10 h-10 mb-3 text-gray-400"
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
                  <p className="mb-2 text-sm text-gray-500">
                    <span className="font-semibold">点击上传</span> 或拖放文件
                  </p>
                  <p className="text-xs text-gray-500">支持MP4、MOV等常见视频格式</p>
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
              <p className="mt-4 text-sm text-gray-500 text-center">
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
                  className="mx-auto border border-gray-300"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
                <div className="absolute top-2 right-2">
                  <button
                    onClick={resetPoints}
                    className="px-3 py-1 bg-gray-600 text-white rounded-lg text-sm"
                  >
                    重选
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-500 text-center mt-2">
                {points.length === 2
                  ? '已选择两点，可以开始处理'
                  : `请在图像上选择${2 - points.length}个点`}
              </p>
            </div>
          )}

          {/* 提交按钮 */}
          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={!file || isProcessing || (!autoMode && points.length !== 2)}
              className={`px-6 py-3 rounded-lg text-white font-medium ${
                !file || isProcessing || (!autoMode && points.length !== 2)
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isProcessing ? '处理中...' : '开始处理'}
            </button>
          </div>
        </div>

        {/* 结果显示 */}
        {result && (
          <div
            className={`bg-white rounded-xl shadow-lg p-8 ${
              result.success ? 'border-green-500' : 'border-red-500'
            } border-2`}
          >
            {result.success ? (
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  处理完成
                </h2>
                <p className="text-lg text-gray-700 mb-6">
                  总移动条纹数: {result.total_fringes}
                </p>
                <button
                  onClick={handleDownload}
                  className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium"
                >
                  下载处理后的视频
                </button>
              </div>
            ) : (
              <div className="text-center">
                <h2 className="text-2xl font-bold text-red-600 mb-4">处理失败</h2>
                <p className="text-gray-700">{result.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
