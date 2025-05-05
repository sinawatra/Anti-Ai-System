"use client"

import type React from "react"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  CheckCircle,
  AlertCircle,
  Clock,
  Info,
  Leaf,
  Camera,
  Palette,
  MapPin,
  Sparkles,
  Wand2,
  Brush,
  Aperture,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface DetectionResultProps {
  isReal: boolean
  confidence: number
  processingTime: number
  reason: string
  indicators: {
    natural?: string[]
    artificial?: string[]
  }
  brandDetected?: string[]
  landscapeFeatures?: string[]
}

export default function DetectionResult({
  isReal,
  confidence,
  processingTime,
  reason,
  indicators = { natural: [], artificial: [] }, // Add a default empty object
  brandDetected = [],
  landscapeFeatures = [],
}: DetectionResultProps) {
  // Make sure indicators always has natural and artificial properties
  const safeIndicators = {
    natural: indicators?.natural || [],
    artificial: indicators?.artificial || [],
  }

  const [activeTab, setActiveTab] = useState("overview")

  // Format confidence to 2 decimal places
  const formattedConfidence = confidence.toFixed(2)

  // Determine if this is a high confidence result
  const isHighConfidence = confidence > 85

  // Group AI indicators by category for better organization
  const groupedArtificialIndicators = {
    style: [] as string[],
    color: [] as string[],
    anatomy: [] as string[],
    technical: [] as string[],
    other: [] as string[],
  }

  // Categorize artificial indicators
  safeIndicators.artificial.forEach((indicator) => {
    const lowerIndicator = indicator.toLowerCase()
    if (lowerIndicator.includes("style") || lowerIndicator.includes("anime") || lowerIndicator.includes("art")) {
      groupedArtificialIndicators.style.push(indicator)
    } else if (
      lowerIndicator.includes("color") ||
      lowerIndicator.includes("neon") ||
      lowerIndicator.includes("rainbow") ||
      lowerIndicator.includes("glow")
    ) {
      groupedArtificialIndicators.color.push(indicator)
    } else if (
      lowerIndicator.includes("face") ||
      lowerIndicator.includes("eye") ||
      lowerIndicator.includes("skin") ||
      lowerIndicator.includes("hair") ||
      lowerIndicator.includes("ear")
    ) {
      groupedArtificialIndicators.anatomy.push(indicator)
    } else if (
      lowerIndicator.includes("pattern") ||
      lowerIndicator.includes("texture") ||
      lowerIndicator.includes("edge") ||
      lowerIndicator.includes("noise")
    ) {
      groupedArtificialIndicators.technical.push(indicator)
    } else {
      groupedArtificialIndicators.other.push(indicator)
    }
  })

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isReal ? (
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <CheckCircle className="mr-1 h-3 w-3" />
                Likely Real
              </Badge>
            ) : (
              <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                <AlertCircle className="mr-1 h-3 w-3" />
                AI Generated
              </Badge>
            )}
            <CardTitle className="text-lg">Confidence: {formattedConfidence}%</CardTitle>
          </div>
          <Badge variant="outline" className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Processed in {processingTime.toFixed(2)} seconds
          </Badge>
        </div>
        <CardDescription>Analysis result for the uploaded image</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="space-y-4 pt-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>{isReal ? "Likely Real" : "AI Generated"}</span>
                <span>{formattedConfidence}%</span>
              </div>
              <Progress
                value={confidence}
                className={cn("h-2", isReal ? "bg-green-100" : "bg-red-100")}
                style={
                  {
                    "--progress-foreground": isReal ? "rgb(22 163 74)" : "rgb(220 38 38)",
                  } as React.CSSProperties
                }
              />
            </div>

            {reason && (
              <div className="mt-4 p-3 bg-slate-50 rounded-md border border-slate-200">
                <div className="flex items-start gap-2">
                  <Info className="h-4 w-4 text-slate-500 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium mb-1">Detection Reason</h4>
                    <p className="text-sm text-slate-600">{reason}</p>
                  </div>
                </div>
              </div>
            )}

            {/* AI Generated Indicators - Only show if AI Generated */}
            {!isReal && safeIndicators.artificial.length > 0 && (
              <div className="mt-4 p-3 bg-red-50 rounded-md border border-red-200">
                <h4 className="text-sm font-medium mb-2 text-red-800 flex items-center gap-2">
                  <Wand2 className="h-4 w-4" />
                  AI Indicators
                </h4>
                <div className="flex flex-wrap gap-2">
                  {safeIndicators.artificial.slice(0, isHighConfidence ? 5 : 3).map((artifact, i) => (
                    <Badge key={i} variant="outline" className="bg-red-100 text-red-700 border-red-200">
                      {artifact}
                    </Badge>
                  ))}
                  {safeIndicators.artificial.length > (isHighConfidence ? 5 : 3) && (
                    <Badge variant="outline" className="bg-red-100 text-red-700 border-red-200">
                      +{safeIndicators.artificial.length - (isHighConfidence ? 5 : 3)} more
                    </Badge>
                  )}
                </div>
                {isHighConfidence && (
                  <p className="text-xs text-red-700 mt-2">
                    High confidence detection based on multiple AI characteristics
                  </p>
                )}
              </div>
            )}

            {/* Brand detection section - Only show if Likely Real */}
            {isReal && brandDetected.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 rounded-md border border-green-200">
                <h4 className="text-sm font-medium mb-2 text-green-800 flex items-center gap-2">
                  <Camera className="h-4 w-4" />
                  Brand Detected
                </h4>
                <div className="flex flex-wrap gap-2">
                  {brandDetected.map((brand, i) => (
                    <Badge key={i} variant="outline" className="bg-green-100 text-green-700 border-green-200">
                      {brand}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-green-700 mt-2">
                  Real-world brand logos are strong indicators of authentic photographs
                </p>
              </div>
            )}

            {/* Landscape features section - Only show if Likely Real */}
            {isReal && landscapeFeatures.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 rounded-md border border-green-200">
                <h4 className="text-sm font-medium mb-2 text-green-800 flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Natural Landscape Features
                </h4>
                <div className="flex flex-wrap gap-2">
                  {landscapeFeatures.slice(0, 3).map((feature, i) => (
                    <Badge key={i} variant="outline" className="bg-green-100 text-green-700 border-green-200">
                      {feature}
                    </Badge>
                  ))}
                  {landscapeFeatures.length > 3 && (
                    <Badge variant="outline" className="bg-green-100 text-green-700 border-green-200">
                      +{landscapeFeatures.length - 3} more
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-green-700 mt-2">
                  Natural landscape features are consistent with authentic outdoor photography
                </p>
              </div>
            )}

            {/* Natural Characteristics - Only show if Likely Real */}
            {isReal && safeIndicators.natural.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 rounded-md border border-green-200">
                <h4 className="text-sm font-medium mb-2 text-green-800 flex items-center gap-2">
                  <Leaf className="h-4 w-4" />
                  Natural Characteristics
                </h4>
                <div className="flex flex-wrap gap-2">
                  {safeIndicators.natural.slice(0, 3).map((element, i) => (
                    <Badge key={i} variant="outline" className="bg-green-100 text-green-700 border-green-200">
                      {element}
                    </Badge>
                  ))}
                  {safeIndicators.natural.length > 3 && (
                    <Badge variant="outline" className="bg-green-100 text-green-700 border-green-200">
                      +{safeIndicators.natural.length - 3} more
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </TabsContent>
          <TabsContent value="details" className="space-y-4 pt-4">
            {/* AI Indicators by Category - Only show if AI Generated */}
            {!isReal && (
              <>
                {groupedArtificialIndicators.style.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Brush className="h-4 w-4 text-red-600" />
                      Artistic Style Indicators
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {groupedArtificialIndicators.style.map((indicator, i) => (
                        <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {groupedArtificialIndicators.color.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Palette className="h-4 w-4 text-red-600" />
                      Color Indicators
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {groupedArtificialIndicators.color.map((indicator, i) => (
                        <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {groupedArtificialIndicators.anatomy.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-red-600" />
                      Anatomical Indicators
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {groupedArtificialIndicators.anatomy.map((indicator, i) => (
                        <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {groupedArtificialIndicators.technical.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Aperture className="h-4 w-4 text-red-600" />
                      Technical Indicators
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {groupedArtificialIndicators.technical.map((indicator, i) => (
                        <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {groupedArtificialIndicators.other.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Info className="h-4 w-4 text-red-600" />
                      Other AI Indicators
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {groupedArtificialIndicators.other.map((indicator, i) => (
                        <Badge key={i} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                          {indicator}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                <div className="p-3 bg-red-50 rounded-md border border-red-200 mt-4">
                  <p className="text-sm text-red-800 font-medium">AI Detection Confidence: {formattedConfidence}%</p>
                  <p className="text-xs text-red-700 mt-1">
                    {confidence > 90
                      ? "Very high confidence in AI detection based on multiple strong indicators."
                      : confidence > 80
                        ? "High confidence in AI detection based on clear indicators."
                        : confidence > 70
                          ? "Moderate confidence in AI detection."
                          : "Lower confidence detection - some AI characteristics detected."}
                  </p>
                </div>
              </>
            )}

            {/* Natural Elements - Only show if Likely Real */}
            {isReal && safeIndicators.natural.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <Leaf className="h-4 w-4 text-green-600" />
                  Natural Elements
                </h4>
                <div className="flex flex-wrap gap-2">
                  {safeIndicators.natural.map((element, i) => (
                    <Badge key={i} variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      {element}
                    </Badge>
                  ))}
                </div>
                <div className="p-3 bg-green-50 rounded-md border border-green-200 mt-4">
                  <p className="text-sm text-green-800 font-medium">Natural Image Confidence: {formattedConfidence}%</p>
                  <p className="text-xs text-green-700 mt-1">
                    {confidence > 90
                      ? "Very high confidence this is a natural image based on multiple indicators."
                      : confidence > 80
                        ? "High confidence this is a natural image."
                        : confidence > 70
                          ? "Moderate confidence this is a natural image."
                          : "Lower confidence - some natural characteristics detected."}
                  </p>
                </div>
              </div>
            )}

            {/* Brand detection details - Only show if Likely Real */}
            {isReal && brandDetected.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <Camera className="h-4 w-4 text-green-600" />
                  Brand Detection
                </h4>
                <div className="p-3 bg-green-50 rounded-md border border-green-200">
                  <p className="text-sm text-green-800">
                    Detected {brandDetected.length > 1 ? "brands" : "brand"}:{" "}
                    <span className="font-medium">{brandDetected.join(", ")}</span>
                  </p>
                  <p className="text-xs text-green-700 mt-2">
                    The presence of real-world brand logos is a strong indicator of an authentic photograph rather than
                    AI-generated content.
                  </p>
                </div>
              </div>
            )}

            {/* Landscape features details - Only show if Likely Real */}
            {isReal && landscapeFeatures.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-green-600" />
                  Landscape Analysis
                </h4>
                <div className="p-3 bg-green-50 rounded-md border border-green-200">
                  <p className="text-sm text-green-800 font-medium mb-1">Detected natural landscape features:</p>
                  <div className="flex flex-wrap gap-2">
                    {landscapeFeatures.map((feature, i) => (
                      <Badge key={i} variant="outline" className="bg-green-100 text-green-700 border-green-200">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-green-700 mt-2">
                    Natural landscapes with consistent lighting, perspective, and environmental features are typically
                    found in real photographs.
                  </p>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
