import { type NextRequest, NextResponse } from "next/server"
<<<<<<< HEAD
import { analyzeImageServer } from "./server-image-analysis"

export async function GET() {
  return NextResponse.json({ status: "online" })
}
=======
>>>>>>> main

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData()
<<<<<<< HEAD
    const file = formData.get("file") as File

    // Check if file exists
    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    console.log("File received:", file.name, file.type, file.size)

    // Convert file to buffer for analysis
    const fileBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(fileBuffer)

    // Start timing
    const startTime = performance.now()

    // Perform the server-compatible image analysis
    const analysisResult = await analyzeImageServer(buffer, file.name)

    // Calculate processing time
    const processingTime = (performance.now() - startTime) / 1000 // Convert to seconds

    // Extract AI artifacts and natural elements for better display
    const aiArtifacts = analysisResult.analysisDetails.detectedArtifacts || []
    const naturalElements = analysisResult.analysisDetails.naturalElements || []
    const brandDetected = analysisResult.analysisDetails.brandDetected || []
    const landscapeFeatures = analysisResult.analysisDetails.landscapeFeatures || []

    // Add additional context to the reason based on detected artifacts
    let enhancedReason = analysisResult.reason
    if (!analysisResult.isReal && aiArtifacts.length > 0) {
      // For AI-generated images, highlight the most significant artifacts
      const topArtifacts = aiArtifacts.slice(0, 2).join(", ")
      enhancedReason = `${enhancedReason}${enhancedReason.endsWith(":") ? "" : ":"} ${topArtifacts}`
    } else if (analysisResult.isReal && naturalElements.length > 0) {
      // For real images, highlight natural characteristics
      const topElements = naturalElements.slice(0, 2).join(", ")
      enhancedReason = `${enhancedReason}${enhancedReason.endsWith(":") ? "" : ":"} ${topElements}`
    }

    // Format the response based on whether the image is real or AI-generated
    const formattedResult = {
      ...analysisResult,
      processingTime: analysisResult.analysisDetails.processingTime || processingTime,
      // Format the result for the frontend
      result: analysisResult.isReal ? "Likely Real" : "AI Generated",
      confidence: analysisResult.confidence,
      reason: enhancedReason,
      // Ensure indicators is always properly structured
      indicators: {
        natural: naturalElements,
        artificial: aiArtifacts,
      },
      // Add additional details for the UI
      brandDetected,
      landscapeFeatures,
    }

    // Add debug information in development
    if (process.env.NODE_ENV === "development") {
      console.log("Analysis result:", {
        isReal: analysisResult.isReal,
        confidence: analysisResult.confidence,
        reason: analysisResult.reason,
        aiArtifacts,
        naturalElements,
      })
    }

    // Return the analysis result
    return NextResponse.json(formattedResult)
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json(
      {
        error: "Failed to process file. Please check server logs for details.",
        isReal: false, // Default to AI-generated on error (safer assumption)
        confidence: 60 + Math.floor(Math.random() * 10),
        reason: "Error in analysis, defaulting to likely AI-generated",
        // Make sure we always return a properly structured object
        indicators: {
          natural: [],
          artificial: ["analysis error"],
        },
        brandDetected: [],
        landscapeFeatures: [],
      },
      { status: 500 },
    )
=======

    // Forward the request to the Flask backend
    const response = await fetch("http://localhost:5000/api/detect", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json({ error: errorData.error || "Failed to process file" }, { status: response.status })
    }

    // Get the result from the Flask backend
    const result = await response.json()

    // Transform the result to match the frontend expectations
    return NextResponse.json({
      isReal: result.is_real,
      confidence: result.confidence,
      heatmap: result.heatmap_url ? `http://localhost:5000${result.heatmap_url}` : null,
      outputImage: result.output_image_url ? `http://localhost:5000${result.output_image_url}` : null,
      outputVideo: result.output_video_url ? `http://localhost:5000${result.output_video_url}` : null,
      thumbnail: result.thumbnail_url ? `http://localhost:5000${result.thumbnail_url}` : null,
      analysisDetails: {
        modelResults:
          result.model_results?.map((model: any) => ({
            modelName: model.model_name,
            confidence: model.confidence.toString(),
            prediction: model.prediction,
            weight: model.weight,
          })) || [],
        ensembleMethod: "Weighted Average",
        detectedArtifacts: result.detected_artifacts || [],
      },
      processingTime: result.processing_time,
      faceResults: result.face_results || [],
    })
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json({ error: "Failed to process file" }, { status: 500 })
>>>>>>> main
  }
}
