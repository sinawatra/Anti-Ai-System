import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData()

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
  }
}
