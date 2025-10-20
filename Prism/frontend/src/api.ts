import axios from 'axios';
export type { PipelineResult, RecordLinkageResult } from './types';
import type { PipelineResult, RecordLinkageResult } from './types';


const apiClient = (baseURL: string) => {
  return axios.create({
    baseURL,
  });
};

const automlApi = apiClient(import.meta.env.VITE_BACKEND_AUTOML_URL || 'http://localhost:8001');
const recordLinkageApi = apiClient(import.meta.env.VITE_BACKEND_RL_URL || 'http://localhost:8000');

// --- AutoML API --- //

export interface AutoMLRequest {
  data_path: string;
  target?: string;
  task: string;
  max_trials: number;
}

export const getDatasetPreview = async (data_path: string) => {
  const response = await automlApi.post('/preview-dataset', { data_path });
  return response.data;
};

export const runAutoML = async (data: AutoMLRequest): Promise<PipelineResult> => {
  const response = await automlApi.post<PipelineResult>('/run', data);
  return response.data;
};

// --- Record Linkage API --- //

export interface LinkageRequest {
  dataset_key: string;
  config_path?: string;
}

export const runRecordLinkage = async (data: LinkageRequest): Promise<RecordLinkageResult> => {
  const response = await recordLinkageApi.post<RecordLinkageResult>('/run', data);
  return response.data;
};
