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
import { motion } from 'framer-motion';
import theme from './theme';
import { getDatasetPreview, runAutoML, runRecordLinkage } from './api';
import type { AutoMLRequest, LinkageRequest, PipelineResult, RecordLinkageResult } from './api';
import DataTable from './components/DataTable';
import ResultsDisplay from './components/ResultsDisplay';

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

  // State for Data Preview
  const [previewData, setPreviewData] = useState<Record<string, any>[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  React.useEffect(() => {
    if (automlDataPath) {
      setPreviewLoading(true);
      setPreviewError(null);
      getDatasetPreview(automlDataPath)
        .then(data => setPreviewData(data))
        .catch(error => {
          setPreviewData(null);
          setPreviewError(error.response?.data?.detail || error.message || 'Could not load preview.');
        })
        .finally(() => setPreviewLoading(false));
    }
  }, [automlDataPath]);

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
          <Grid xs={12} md={4}>
            <motion.div initial={{ opacity: 0, x: -50 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5 }}>
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
          </motion.div>
        </Grid>

          {/* Main Content */}
          <Grid xs={12} md={8}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.2 }}>
              <Box display="flex" flexDirection="column" gap={2}>
                {/* Data Preview Section */}
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Dataset Preview</Typography>
                {previewLoading && <CircularProgress />}
                {previewError && <Alert severity="error">{previewError}</Alert>}
                {previewData && <DataTable data={previewData} />}
              </Paper>

              {/* Results Section */}
              <ResultsDisplay
                automlResult={automlResult}
                rlResult={rlResult}
                automlError={automlError}
                rlError={rlError}
                automlLoading={automlLoading}
                rlLoading={rlLoading}
              />
            </Box>
          </motion.div>
        </Grid>
        </Grid>
      </Box>
    </ThemeProvider>
  );
}

export default App;