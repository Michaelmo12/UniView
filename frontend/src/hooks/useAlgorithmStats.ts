import { useState, useEffect } from "react";
import { apiRequest } from "../services/api/client";

interface StageTimings {
  detection: number;
  features: number;
  fusion: number;
  reconstruction: number;
  tracking: number;
  total: number;
}

interface AlgorithmStatus {
  active_drones: number;
  active_tracks: number;
  server_fps: number;
  system_status: "Optimal" | "Warning" | "Critical";
  avg_pipeline_latency_ms: number;
  avg_confidence: number;
  stage_timings_ms: StageTimings;
}

interface UseAlgorithmStatsResult {
  data: AlgorithmStatus | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useAlgorithmStats(): UseAlgorithmStatsResult {
  const [data, setData] = useState<AlgorithmStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiRequest<AlgorithmStatus>("/algorithm/status");
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return { data, loading, error, refetch: fetchStatus };
}
