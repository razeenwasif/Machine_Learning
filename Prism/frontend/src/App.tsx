import React, { useState } from 'react';
import {
  CssBaseline,
  ThemeProvider,
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Select,
  MenuItem,
  Slider,
} from '@mui/material';
import theme from './theme';
import { runAutoML, runRecordLinkage, AutoMLRequest, LinkageRequest } from './api';
import { PipelineResult, RecordLinkageResult } from './types';

function App() {
  // State for AutoML
  const [automlDataPath, setAutomlDataPath] = useState('/workspace/datasets/customer_churn.csv');
  const [automlTarget, setAutomlTarget] = useState('churned');
  const [automlTask, setAutomlTask] = useState('auto');
  const [automlMaxTrials, setAutomlMaxTrials] = useState(20);
  const [automlResult, setAutomlResult] = useState<PipelineResult | null>(null);
  const [automlLoading, setAutomlLoading] = useState(false);
  const [automlError, setAutomlError] = useState<string | null>(null);

  // State for Record Linkage
  const [rlDatasetKey, setRlDatasetKey] = useState('assignment_datasets');
  const [rlConfigPath, setRlConfigPath] = useState('recordLinkage/config/pipeline.toml');
  const [rlResult, setRlResult] = useState<RecordLinkageResult | null>(null);
  const [rlLoading, setRlLoading] = useState(false);
  const [rlError, setRlError] = useState<string | null>(null);

  const handleRunAutoML = async () => {
    setAutomlLoading(true);
    setAutomlError(null);
    setAutomlResult(null);
    try {
      const requestData: AutoMLRequest = {
        data_path: automlDataPath,
        target: automlTarget,
        task: automlTask,
        max_trials: automlMaxTrials,
      };
      const result = await runAutoML(requestData);
      setAutomlResult(result);
    } catch (error: any) {
      setAutomlError(error.response?.data?.detail || error.message || 'An unknown error occurred.');
    } finally {
      setAutomlLoading(false);
    }
  };

  const handleRunRecordLinkage = async () => {
    setRlLoading(true);
    setRlError(null);
    setRlResult(null);
    try {
      const requestData: LinkageRequest = {
        dataset_key: rlDatasetKey,
        config_path: rlConfigPath,
      };
      const result = await runRecordLinkage(requestData);
      setRlResult(result);
    } catch (error: any) {
      setRlError(error.response?.data?.detail || error.message || 'An unknown error occurred.');
    } finally {
      setRlLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, p: 2 }}>
        <Grid container spacing={2}>
          {/* Sidebar */}
          <Grid item xs={12} md={4}>
            <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
              <Typography variant="h5" gutterBottom>Configuration</Typography>

              {/* AutoML Section */}
              <Box mb={4}>
                <Typography variant="h6">AutoML Pipeline</Typography>
                <TextField
                  label="Dataset Path"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  size="small"
                  value={automlDataPath}
                  onChange={(e) => setAutomlDataPath(e.target.value)}
                />
                <TextField
                  label="Target Column"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  size="small"
                  value={automlTarget}
                  onChange={(e) => setAutomlTarget(e.target.value)}
                />
                <Select
                  value={automlTask}
                  onChange={(e) => setAutomlTask(e.target.value)}
                  fullWidth
                  size="small"
                >
                  <MenuItem value="auto">Auto</MenuItem>
                  <MenuItem value="classification">Classification</MenuItem>
                  <MenuItem value="regression">Regression</MenuItem>
                  <MenuItem value="clustering">Clustering</MenuItem>
                </Select>
                <Typography gutterBottom sx={{ mt: 2 }}>Max Trials: {automlMaxTrials}</Typography>
                <Slider
                  value={automlMaxTrials}
                  onChange={(_, value) => setAutomlMaxTrials(value as number)}
                  aria-labelledby="automl-max-trials-slider"
                  valueLabelDisplay="auto"
                  step={5}
                  marks
                  min={5}
                  max={50}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleRunAutoML}
                  disabled={automlLoading}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  {automlLoading ? <CircularProgress size={24} /> : 'Run AutoML'}
                </Button>
              </Box>

              {/* Record Linkage Section */}
              <Box>
                <Typography variant="h6">Record Linkage</Typography>
                <TextField
                  label="Dataset Key"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  size="small"
                  value={rlDatasetKey}
                  onChange={(e) => setRlDatasetKey(e.target.value)}
                />
                <TextField
                  label="Config Path"
                  variant="outlined"
                  fullWidth
                  margin="normal"
                  size="small"
                  value={rlConfigPath}
                  onChange={(e) => setRlConfigPath(e.target.value)}
                />
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={handleRunRecordLinkage}
                  disabled={rlLoading}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  {rlLoading ? <CircularProgress size={24} /> : 'Run Record Linkage'}
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Main Content */}
          <Grid item xs={12} md={8}>
            <ResultsDisplay
              automlResult={automlResult}
              rlResult={rlResult}
              automlError={automlError}
              rlError={rlError}
              automlLoading={automlLoading}
              rlLoading={rlLoading}
            />
          </Grid>
        </Grid>
      </Box>
    </ThemeProvider>
  );
}

export default App;