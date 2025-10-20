// TypeScript interfaces for API data structures

export interface AnalysisReport {
  original_preview: Record<string, any>[];
  missing_by_column: Record<string, number>;
  notes: string[];
  numeric_profiles: any[]; // Define more strictly if possible
  categorical_profiles: any[]; // Define more strictly if possible
  target_summary: any; // Define more strictly if possible
  correlation_pairs: any[]; // Define more strictly if possible
}

export interface CleaningReport {
  applied_steps: string[];
  dropped_columns: string[];
  filled_columns: Record<string, string>;
  outlier_treatments: Record<string, string>;
}

export interface PipelineResult {
  task: string;
  model_name: string;
  metrics: Record<string, number>;
  best_config: Record<string, any>;
  hpo_score: number;
  analysis_report: AnalysisReport;
  cleaning_report: CleaningReport;
}

export interface RecordLinkageResult {
  dataset_key: string;
  dataset_a_path: string;
  dataset_b_path: string;
  truth_path?: string;
  output_path: string;
  id_column: string;
  attributes: string[];
  match_count: number;
  non_match_count: number;
  true_match_count: number;
  candidate_pairs: number;
  filtered_pairs: number;
  blocking_metrics: Record<string, number>;
  linkage_metrics: Record<string, number>;
  runtime: Record<string, number>;
  analysis_a: AnalysisReport;
  analysis_b: AnalysisReport;
}
