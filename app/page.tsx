"use client"

import type React from "react"

<<<<<<< HEAD
import { useState, useEffect } from "react"
import { Upload, Info, AlertCircle, Layers } from "lucide-react"
=======
import { useState } from "react"
import { Upload, Info, AlertCircle, BarChart3, Layers, Clock, Video } from "lucide-react"
>>>>>>> main
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
<<<<<<< HEAD
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import DetectionResultComponent from "@/components/detection-result"

type DetectionResultType = {
  isReal: boolean
  confidence: number
  processingTime?: number
  analysisDetails?: {
    modelResults?: any[]
    detectedArtifacts: string[]
    naturalElements?: string[]
    detectedSubject?: string
    humanDetected?: boolean
    realWorldIndicators?: string[]
    reason?: string
    brandDetected?: string[]
    landscapeFeatures?: string[]
=======
import { Badge } from "@/components/ui/badge"

type AnalysisDetail = {
  modelName: string
  confidence: string
  prediction: string
  weight: number
}

type FaceResult = {
  is_real: boolean
  confidence: number
  bbox: number[]
}

type DetectionResult = {
  isReal: boolean
  confidence: number
  heatmap?: string
  outputImage?: string
  outputVideo?: string
  thumbnail?: string
  processingTime?: number
  faceResults?: FaceResult[]
  analysisDetails?: {
    modelResults: AnalysisDetail[]
    ensembleMethod: string
    detectedArtifacts: string[]
>>>>>>> main
  }
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
<<<<<<< HEAD
  const [result, setResult] = useState<DetectionResultType | null>(null)
  const [activeTab, setActiveTab] = useState("image")
  const [advancedMode, setAdvancedMode] = useState(true)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<"unknown" | "online" | "offline">("unknown")

  // Check backend status on component mount
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch("/api/detect", {
          method: "GET",
          cache: "no-store",
        })

        if (response.ok) {
          const data = await response.json()
          setBackendStatus(data.status === "online" ? "online" : "offline")
        } else {
          setBackendStatus("offline")
        }
      } catch (error) {
        console.error("Error checking backend status:", error)
        setBackendStatus("offline")
      }
    }

    checkBackendStatus()
  }, [])
=======
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [activeTab, setActiveTab] = useState("image")
  const [advancedMode, setAdvancedMode] = useState(true)
  const [progress, setProgress] = useState(0)
>>>>>>> main

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

<<<<<<< HEAD
    // Clear previous errors and results
    setError(null)
    setResult(null)
=======
>>>>>>> main
    setFile(selectedFile)

    // Create preview for image files
    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreview(e.target?.result as string)
      }
      reader.readAsDataURL(selectedFile)
    } else if (selectedFile.type.startsWith("video/")) {
      // Create a video thumbnail
      const videoElement = document.createElement("video")
      videoElement.preload = "metadata"

      videoElement.onloadedmetadata = () => {
        // Set the current time to 1 second or the middle of the video
        videoElement.currentTime = Math.min(1, videoElement.duration / 2)
      }

      videoElement.onloadeddata = () => {
        // Create a canvas to capture the frame
        const canvas = document.createElement("canvas")
        canvas.width = videoElement.videoWidth
        canvas.height = videoElement.videoHeight

        // Draw the video frame on the canvas
        const ctx = canvas.getContext("2d")
        if (ctx) {
          ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height)

          // Convert the canvas to a data URL and set as preview
          const thumbnailUrl = canvas.toDataURL("image/jpeg")
          setPreview(thumbnailUrl)
        }
      }

      // Set the video source to the file
      videoElement.src = URL.createObjectURL(selectedFile)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    setIsLoading(true)
    setProgress(0)
<<<<<<< HEAD
    setError(null)
    setResult(null)

    // Create a more realistic progress simulation for 5-6 seconds
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          return 90 // Hold at 90% until actual completion
        }

        // Calculate appropriate increment for 5-6 second duration
        // Aim to reach ~90% in about 5 seconds
        const increment =
          prev < 30
            ? 5
            : // Fast at start
              prev < 60
              ? 3
              : // Medium in middle
                prev < 80
                ? 2
                : // Slower approaching end
                  1 // Very slow at end

        return prev + increment
      })
    }, 250) // Update every 250ms for smoother progress
=======

    // Create a progress simulation
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + 5
      })
    }, 500)
>>>>>>> main

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to process file")
      }

      const data = await response.json()
<<<<<<< HEAD
      console.log("Response data:", data)
      setResult(data)
    } catch (error) {
      console.error("Error:", error)
      setError(error instanceof Error ? error.message : "Failed to process file")
=======
      setResult(data)
    } catch (error) {
      console.error("Error:", error)
      alert(`Error: ${error instanceof Error ? error.message : "Failed to process file"}`)
>>>>>>> main
    } finally {
      setIsLoading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setProgress(0)
<<<<<<< HEAD
    setError(null)
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-24 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2 dark:text-white">Enhanced DeepFake Detector</h1>
          <p className="text-gray-600 dark:text-gray-300">
            Upload an image or video to detect if it was generated by AI using advanced analysis
=======
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-24 bg-gray-50">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Enhanced DeepFake Detector</h1>
          <p className="text-gray-600">
            Upload an image or video to detect if it was generated by AI using FaceForensics++ model
>>>>>>> main
          </p>
          <div className="mt-4 flex justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAdvancedMode(!advancedMode)}
              className="flex items-center gap-2"
            >
              <Layers className="h-4 w-4" />
              {advancedMode ? "Simple Mode" : "Advanced Mode"}
            </Button>
          </div>
        </div>

<<<<<<< HEAD
        {backendStatus === "offline" && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Backend Service Unavailable</AlertTitle>
            <AlertDescription>
              The detection service is currently offline. Please ensure the API route is working properly.
            </AlertDescription>
          </Alert>
        )}

=======
>>>>>>> main
        <Tabs defaultValue="image" className="mb-8" onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="image">Image Detection</TabsTrigger>
            <TabsTrigger value="video">Video Detection</TabsTrigger>
          </TabsList>
          <TabsContent value="image" className="mt-4">
            <Card className="p-6">
<<<<<<< HEAD
              <h2 className="text-xl font-semibold mb-4 dark:text-white">Upload an Image</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Our enhanced AI will analyze the image using multiple detection techniques.
              </p>
              {advancedMode && (
                <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg mt-2">
                  <h3 className="font-medium text-blue-700 dark:text-blue-300 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Advanced Detection Enabled
                  </h3>
                  <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">
                    Using enhanced natural image recognition with advanced pattern analysis
=======
              <h2 className="text-xl font-semibold mb-4">Upload an Image</h2>
              <p className="text-gray-600 mb-4">
                Our enhanced AI will analyze the image using the FaceForensics++ model.
              </p>
              {advancedMode && (
                <div className="bg-blue-50 p-4 rounded-lg mt-2">
                  <h3 className="font-medium text-blue-700 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Advanced Detection Enabled
                  </h3>
                  <p className="text-sm text-blue-600 mt-1">
                    Using FaceForensics++ model with face detection and manipulation analysis
>>>>>>> main
                  </p>
                </div>
              )}
            </Card>
          </TabsContent>
          <TabsContent value="video" className="mt-4">
            <Card className="p-6">
<<<<<<< HEAD
              <h2 className="text-xl font-semibold mb-4 dark:text-white">Upload a Video</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Our AI will analyze frames of the video to detect deepfakes.
              </p>
              {advancedMode && (
                <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg mt-2">
                  <h3 className="font-medium text-blue-700 dark:text-blue-300 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Advanced Video Analysis Enabled
                  </h3>
                  <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">
=======
              <h2 className="text-xl font-semibold mb-4">Upload a Video</h2>
              <p className="text-gray-600 mb-4">Our AI will analyze frames of the video to detect deepfakes.</p>
              {advancedMode && (
                <div className="bg-blue-50 p-4 rounded-lg mt-2">
                  <h3 className="font-medium text-blue-700 flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    Advanced Video Analysis Enabled
                  </h3>
                  <p className="text-sm text-blue-600 mt-1">
>>>>>>> main
                    Using temporal consistency analysis and frame-by-frame deepfake detection
                  </p>
                </div>
              )}
            </Card>
          </TabsContent>
        </Tabs>

<<<<<<< HEAD
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {!file && (
            <div
              className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
              onClick={() => document.getElementById("file-upload")?.click()}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Click to upload or drag and drop</p>
              <p className="text-xs text-gray-500 dark:text-gray-500">
=======
        <form onSubmit={handleSubmit} className="space-y-6">
          {!file && (
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => document.getElementById("file-upload")?.click()}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">Click to upload or drag and drop</p>
              <p className="text-xs text-gray-500">
>>>>>>> main
                {activeTab === "image" ? "PNG, JPG, WEBP up to 10MB" : "MP4, MOV up to 50MB"}
              </p>
              <input
                id="file-upload"
                name="file-upload"
                type="file"
                className="sr-only"
                accept={activeTab === "image" ? "image/*" : "video/*"}
                onChange={handleFileChange}
              />
            </div>
          )}

          {preview && (
            <div className="mt-4 relative">
              <div className="flex justify-between items-center mb-2">
<<<<<<< HEAD
                <h3 className="font-medium dark:text-white">Preview</h3>
=======
                <h3 className="font-medium">Preview</h3>
>>>>>>> main
                <Button variant="ghost" size="sm" onClick={resetForm}>
                  Change file
                </Button>
              </div>
<<<<<<< HEAD
              <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
=======
              <div className="rounded-lg overflow-hidden border border-gray-200">
>>>>>>> main
                <img
                  src={preview || "/placeholder.svg"}
                  alt="File preview"
                  className="w-full h-auto max-h-[400px] object-contain"
                />
              </div>
              <div className="mt-4">
<<<<<<< HEAD
                <Button type="submit" className="w-full" disabled={isLoading || backendStatus === "offline"}>
=======
                <Button type="submit" className="w-full" disabled={isLoading}>
>>>>>>> main
                  {isLoading ? "Analyzing..." : "Detect Fake"}
                </Button>
              </div>
            </div>
          )}
        </form>

        {isLoading && (
          <div className="mt-8 space-y-4">
<<<<<<< HEAD
            <p className="text-center font-medium dark:text-white">
              {advancedMode ? "Running advanced image analysis..." : "Analyzing..."}
=======
            <p className="text-center font-medium">
              {advancedMode ? "Running analysis with FaceForensics++..." : "Analyzing..."}
>>>>>>> main
            </p>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {result && (
<<<<<<< HEAD
          <div className="mt-8">
            <DetectionResultComponent
              isReal={result.isReal}
              confidence={result.confidence}
              processingTime={result.processingTime || 0}
              analysisDetails={{
                detectedArtifacts: result.analysisDetails?.detectedArtifacts || [],
                naturalElements: result.analysisDetails?.naturalElements || [],
                detectedSubject: result.analysisDetails?.detectedSubject || null,
                humanDetected: result.analysisDetails?.humanDetected || false,
                realWorldIndicators: result.analysisDetails?.realWorldIndicators || [],
                reason: result.analysisDetails?.reason || undefined,
                brandDetected: result.analysisDetails?.brandDetected || [],
                landscapeFeatures: result.analysisDetails?.landscapeFeatures || [],
              }}
            />
=======
          <Card className="mt-8 p-6">
            <div className="flex items-center gap-4 mb-4">
              {result.isReal ? (
                <div className="bg-green-100 p-3 rounded-full">
                  <Info className="h-6 w-6 text-green-600" />
                </div>
              ) : (
                <div className="bg-red-100 p-3 rounded-full">
                  <AlertCircle className="h-6 w-6 text-red-600" />
                </div>
              )}
              <div>
                <h3 className="text-xl font-bold">{result.isReal ? "Likely Real" : "Likely AI-Generated"}</h3>
                <p className="text-gray-600">Confidence: {result.confidence.toFixed(2)}%</p>
                {result.processingTime && (
                  <p className="text-xs text-gray-500 flex items-center gap-1 mt-1">
                    <Clock className="h-3 w-3" /> Processed in {result.processingTime.toFixed(2)} seconds
                  </p>
                )}
              </div>
            </div>

            {/* Display output image or video */}
            {result.outputImage && (
              <div className="mt-6">
                <h4 className="font-medium mb-2">Analysis Result</h4>
                <div className="border rounded-lg overflow-hidden">
                  <img src={result.outputImage || "/placeholder.svg"} alt="Analysis result" className="w-full h-auto" />
                </div>
              </div>
            )}

            {result.outputVideo && (
              <div className="mt-6">
                <h4 className="font-medium mb-2 flex items-center gap-2">
                  <Video className="h-4 w-4" /> Video Analysis
                </h4>
                <div className="border rounded-lg overflow-hidden">
                  {result.thumbnail && (
                    <div className="relative">
                      <img
                        src={result.thumbnail || "/placeholder.svg"}
                        alt="Video thumbnail"
                        className="w-full h-auto"
                      />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <a
                          href={result.outputVideo}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="bg-black bg-opacity-70 text-white p-3 rounded-full hover:bg-opacity-90 transition-all"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                          </svg>
                        </a>
                      </div>
                    </div>
                  )}
                </div>
                <p className="text-sm text-gray-500 mt-2">Click to view the analyzed video</p>
              </div>
            )}

            {advancedMode && result.analysisDetails && (
              <div className="mt-6">
                <h4 className="font-medium mb-3 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Model Analysis Breakdown
                </h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    {result.analysisDetails.modelResults.map((model, index) => (
                      <div key={index} className="bg-white p-3 rounded-lg border">
                        <h5 className="font-medium text-sm">{model.modelName}</h5>
                        <div className="mt-2 flex justify-between items-center">
                          <Badge variant={model.prediction === "Real" ? "success" : "destructive"}>
                            {model.prediction}
                          </Badge>
                          <span className="text-sm font-medium">{Number.parseFloat(model.confidence).toFixed(1)}%</span>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">Weight: {model.weight * 100}%</p>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-gray-600">
                    Ensemble Method: <span className="font-medium">{result.analysisDetails.ensembleMethod}</span>
                  </p>
                </div>
              </div>
            )}

            {!result.isReal && result.heatmap && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Manipulation Heatmap</h4>
                <div className="border rounded-lg overflow-hidden">
                  <img
                    src={result.heatmap || "/placeholder.svg"}
                    alt="Manipulation heatmap"
                    className="w-full h-auto"
                  />
                </div>
                <p className="text-sm text-gray-500 mt-2">
                  Red areas indicate potential manipulations or AI-generated content
                </p>
              </div>
            )}

            {advancedMode && !result.isReal && result.analysisDetails?.detectedArtifacts.length > 0 && (
              <div className="mt-4">
                <h4 className="font-medium mb-2">Detected Artifacts</h4>
                <ul className="list-disc pl-5 space-y-1">
                  {result.analysisDetails.detectedArtifacts.map((artifact, index) => (
                    <li key={index} className="text-sm text-gray-700">
                      {artifact}
                    </li>
                  ))}
                </ul>
              </div>
            )}
>>>>>>> main

            <Accordion type="single" collapsible className="mt-6">
              <AccordionItem value="how-it-works">
                <AccordionTrigger className="text-sm font-medium">How it works</AccordionTrigger>
                <AccordionContent>
<<<<<<< HEAD
                  <div className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
                    <p>This detector uses multiple advanced analysis techniques:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Enhanced natural image recognition for real photographs</li>
                      <li>Advanced pattern analysis for AI-generated content</li>
                      <li>Facial feature analysis for human subjects</li>
                      <li>Color profile and texture consistency evaluation</li>
                      <li>Natural subject recognition for animals, humans, and landscapes</li>
                      <li>Metadata analysis for photographic indicators</li>
                      <li>Brand logo detection for authentic photographs</li>
                      <li>Natural landscape feature recognition</li>
                    </ul>
                    <p>
                      The system combines these approaches to accurately distinguish between real images and
                      AI-generated content with high confidence.
=======
                  <div className="text-sm text-gray-600 space-y-2">
                    <p>This detector uses the FaceForensics++ model to analyze media:</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Face detection using dlib to locate faces in the image/video</li>
                      <li>Xception CNN architecture trained on the FaceForensics++ dataset</li>
                      <li>Frame-by-frame analysis for videos to detect temporal inconsistencies</li>
                      <li>Heatmap generation to highlight potentially manipulated regions</li>
                    </ul>
                    <p>
                      The model has been trained on thousands of real and fake images to identify subtle patterns
                      indicative of AI manipulation.
>>>>>>> main
                    </p>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
<<<<<<< HEAD
          </div>
=======
          </Card>
>>>>>>> main
        )}
      </div>
    </main>
  )
}
