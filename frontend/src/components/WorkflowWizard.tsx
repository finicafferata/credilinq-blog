import React, { useState, useEffect } from 'react';
import { ChevronRightIcon, CheckIcon } from '@heroicons/react/24/outline';

interface WorkflowStep {
  id: 'planner' | 'researcher' | 'writer' | 'editor' | 'image' | 'seo' | 'social_media';
  title: string;
  description: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
}

interface WorkflowState {
  workflow_id: string;
  current_step: WorkflowStep['id'];
  progress: number;
  outline?: string[];
  research?: Record<string, any>;
  content?: string;
  editor_feedback?: any;
  images?: any[];
  seo_analysis?: any;
  social_posts?: any;
  blog_title: string;
  company_context: string;
  mode?: 'quick' | 'advanced' | 'template';
}

type WorkflowMode = 'quick' | 'advanced' | 'template';

interface ModeOption {
  id: WorkflowMode;
  title: string;
  description: string;
  icon: string;
  features: string[];
}

const WorkflowWizard: React.FC = () => {
  const [workflowState, setWorkflowState] = useState<WorkflowState | null>(null);
  const [selectedMode, setSelectedMode] = useState<WorkflowMode | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const modeOptions: ModeOption[] = [
    {
      id: 'quick',
      title: 'Quick Mode',
      description: 'Fast blog creation with AI assistance',
      icon: '⚡',
      features: ['AI-powered writing', 'Basic SEO optimization', 'Ready in minutes']
    },
    {
      id: 'advanced',
      title: 'Advanced Workflow',
      description: 'Complete multi-agent content creation process',
      icon: '🎯',
      features: ['Research & planning', 'Multi-step review', 'Images & social media', 'Full SEO analysis']
    },
    {
      id: 'template',
      title: 'Template Mode',
      description: 'Start with pre-built content templates',
      icon: '📋',
      features: ['Industry templates', 'Customizable structure', 'Guided workflow']
    }
  ];

  const getStepsForMode = (mode: WorkflowMode): WorkflowStep[] => {
    const allSteps: WorkflowStep[] = [
    {
      id: 'planner',
      title: 'Planning',
      description: 'Create structured content outline',
      status: 'pending'
    },
    {
      id: 'researcher',
      title: 'Research',
      description: 'Research each section of the outline',
      status: 'pending'
    },
    {
      id: 'writer',
      title: 'Writing',
      description: 'Generate content based on research',
      status: 'pending'
    },
    {
      id: 'editor',
      title: 'Review',
      description: 'Review and approve final content',
      status: 'pending'
    },
    {
      id: 'image',
      title: 'Images',
      description: 'Generate images for content',
      status: 'pending'
    },
    {
      id: 'seo',
      title: 'SEO',
      description: 'Optimize content for search engines',
      status: 'pending'
    },
    {
      id: 'social_media',
      title: 'Social Media',
      description: 'Adapt content for different platforms',
      status: 'pending'
    }
    ];

    switch (mode) {
      case 'quick':
        return allSteps.filter(step => ['writer', 'editor'].includes(step.id));
      case 'template':
        return allSteps.filter(step => ['planner', 'writer', 'editor', 'seo'].includes(step.id));
      case 'advanced':
      default:
        return allSteps;
    }
  };

  const steps = workflowState ? getStepsForMode(workflowState.mode || 'advanced') : [];

  const startWorkflow = async (title: string, context: string, mode: WorkflowMode = 'advanced') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/workflow-fixed/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, company_context: context, mode })
      });
      
      if (!response.ok) throw new Error('Error starting workflow');
      
      const data = await response.json();
      console.log('Initial workflow response:', data);
      setWorkflowState(data);
      
      // Start polling for progress updates
      if (data.workflow_id) {
        startProgressPolling(data.workflow_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsLoading(false);
    }
  };

  const startProgressPolling = (workflowId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/workflow-fixed/status/${workflowId}`);
        if (!response.ok) {
          console.error('Failed to fetch workflow status');
          return;
        }
        
        const data = await response.json();
        console.log('Progress update:', data);
        setWorkflowState(data);
        
        // Stop polling when workflow is complete or failed
        if (data.progress >= 100 || data.status === 'failed' || data.status === 'completed') {
          clearInterval(pollInterval);
          setIsLoading(false);
          console.log('Workflow completed, stopping polling');
        }
      } catch (err) {
        console.error('Error polling workflow status:', err);
      }
    }, 2000); // Poll every 2 seconds
    
    // Clean up after 5 minutes to avoid infinite polling
    setTimeout(() => {
      clearInterval(pollInterval);
      setIsLoading(false);
    }, 300000);
  };

  // Removed manual execution - workflow now runs automatically

  const getStepStatus = (stepId: WorkflowStep['id']): WorkflowStep['status'] => {
    if (!workflowState) return 'pending';
    
    // If workflow is completed (progress 100), all steps are completed
    if (workflowState.progress === 100) return 'completed';
    
    // If it's the current step, it's in progress
    if (workflowState.current_step === stepId) return 'in-progress';
    
    // Check if this step has been completed based on workflow progression
    // Define step order for different modes
    const stepOrder = {
      'quick': ['writer', 'editor'],
      'template': ['planner', 'writer', 'editor', 'seo'],
      'advanced': ['planner', 'researcher', 'writer', 'editor', 'image', 'seo', 'social_media']
    };
    
    const mode = workflowState.mode || 'advanced';
    const currentModeSteps = stepOrder[mode as keyof typeof stepOrder] || stepOrder.advanced;
    const currentStepIndex = currentModeSteps.indexOf(workflowState.current_step);
    const thisStepIndex = currentModeSteps.indexOf(stepId);
    
    // If this step comes before the current step, it's completed
    if (thisStepIndex >= 0 && thisStepIndex < currentStepIndex) {
      return 'completed';
    }
    
    // If this step is not in the current mode's workflow, mark as pending
    if (thisStepIndex === -1) {
      return 'pending';
    }
    
    return 'pending';
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Intelligent Content Generator
        </h1>
        <p className="text-gray-600">
          Create professional content using specialized agents
        </p>
      </div>

      {/* Mode Selection */}
      {!selectedMode && !workflowState && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Choose Content Creation Mode</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {modeOptions.map((mode) => (
              <div
                key={mode.id}
                onClick={() => setSelectedMode(mode.id)}
                className="border border-gray-200 rounded-lg p-6 cursor-pointer hover:border-blue-500 hover:shadow-md transition-all"
              >
                <div className="text-center mb-4">
                  <div className="text-3xl mb-2">{mode.icon}</div>
                  <h3 className="text-lg font-semibold text-gray-900">{mode.title}</h3>
                  <p className="text-sm text-gray-600 mt-2">{mode.description}</p>
                </div>
                <ul className="space-y-2">
                  {mode.features.map((feature, index) => (
                    <li key={index} className="flex items-center text-sm text-gray-700">
                      <CheckIcon className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>
                <button className="w-full mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors">
                  Select {mode.title}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input inicial */}
      {selectedMode && !workflowState && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Start New Content - {modeOptions.find(m => m.id === selectedMode)?.title}</h2>
            <button 
              onClick={() => setSelectedMode(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ← Back to mode selection
            </button>
          </div>
          <WorkflowInput onStart={(title, context) => startWorkflow(title, context, selectedMode)} isLoading={isLoading} />
        </div>
      )}

      {/* Workflow Progress */}
      {workflowState && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Workflow Progress</h2>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                {modeOptions.find(m => m.id === workflowState.mode)?.title || 'Advanced Mode'}
              </span>
            </div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex-1 bg-gray-200 rounded-full h-2 mr-4">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${workflowState.progress}%` }}
                />
              </div>
              <span className="text-sm font-medium text-gray-600">
                {workflowState.progress}%
              </span>
            </div>
          </div>

          {/* Workflow Steps */}
          <div className="space-y-4">
            {steps.map((step, index) => {
              const stepStatus = getStepStatus(step.id);
              const isStepActive = workflowState.current_step === step.id;
              console.log(`Step ${step.id}: status=${stepStatus}, isActive=${isStepActive}, current_step=${workflowState.current_step}`);
              
              return (
                <WorkflowStep
                  key={step.id}
                  step={step}
                  status={stepStatus}
                  isActive={isStepActive}
                  data={workflowState}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <p className="text-red-800">{error}</p>
        </div>
      )}
    </div>
  );
};

// Componente para input inicial
const WorkflowInput: React.FC<{
  onStart: (title: string, context: string) => void;
  isLoading: boolean;
}> = ({ onStart, isLoading }) => {
  const [title, setTitle] = useState('');
  const [context, setContext] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (title.trim() && context.trim()) {
      onStart(title.trim(), context.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Content Title
        </label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Ex: Complete guide to digital marketing"
          required
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Company Context
        </label>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          placeholder="Describe your company, target audience, communication tone..."
          required
        />
      </div>
      
      <button
        type="submit"
        disabled={isLoading || !title.trim() || !context.trim()}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? 'Starting...' : 'Start Creating'}
      </button>
    </form>
  );
};

// Componente para cada paso del workflow
const WorkflowStep: React.FC<{
  step: WorkflowStep;
  status: WorkflowStep['status'];
  isActive: boolean;
  data: WorkflowState;
}> = ({ step, status, isActive, data }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckIcon className="w-5 h-5 text-green-600" />;
      case 'in-progress':
        return <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      default:
        return <ChevronRightIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200';
      case 'in-progress':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            {getStatusIcon()}
          </div>
          <div>
            <h3 className="text-lg font-medium text-gray-900">{step.title}</h3>
            <p className="text-sm text-gray-600">{step.description}</p>
          </div>
        </div>
        
        <div className="flex space-x-2">
          {status === 'in-progress' && (
            <div className="flex items-center space-x-2 text-sm text-blue-600 font-medium px-3 py-1 bg-blue-50 rounded">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
              <span>Processing...</span>
            </div>
          )}
          {status === 'completed' && (
            <div className="text-sm text-green-600 font-medium px-3 py-1 bg-green-50 rounded">
              ✓ Completed
            </div>
          )}
          {status === 'pending' && isActive && (
            <div className="flex items-center space-x-2 text-sm text-yellow-600 font-medium px-3 py-1 bg-yellow-50 rounded">
              <div className="w-2 h-2 bg-yellow-600 rounded-full animate-bounce"></div>
              <span>Queued...</span>
            </div>
          )}
        </div>
      </div>

      {/* Show step data if available */}
      {status === 'completed' && (
        <div className="mt-4 p-3 bg-white rounded border">
          <StepData stepId={step.id} data={data} />
        </div>
      )}
    </div>
  );
};

// Componente para mostrar datos de cada paso
const StepData: React.FC<{ stepId: string; data: WorkflowState }> = ({ stepId, data }) => {
  switch (stepId) {
    case 'planner':
      return (
        <div>
          <h4 className="font-medium mb-2">Generated Outline:</h4>
          <ul className="list-disc list-inside space-y-1">
            {data.outline?.map((section, index) => (
              <li key={index} className="text-sm text-gray-700">{section}</li>
            ))}
          </ul>
        </div>
      );
    
    case 'researcher':
      return (
        <div>
          <h4 className="font-medium mb-2">Research Completed:</h4>
          <p className="text-sm text-gray-600">
            {Object.keys(data.research || {}).length} sections researched
          </p>
        </div>
      );
    
    case 'writer':
      return (
        <div>
          <h4 className="font-medium mb-2">Generated Content:</h4>
          <div className="text-sm text-gray-700 max-h-32 overflow-y-auto mb-3">
            {data.content?.substring(0, 200)}...
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => window.open(`data:text/plain;charset=utf-8,${encodeURIComponent(data.content || '')}`, '_blank')}
              className="bg-green-600 text-white px-3 py-1 rounded text-xs hover:bg-green-700"
            >
              Download
            </button>
            <button
              onClick={() => navigator.clipboard.writeText(data.content || '')}
              className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700"
            >
              Copy
            </button>
            <button
              onClick={() => {/* TODO: Implementar modal */}}
              className="bg-purple-600 text-white px-3 py-1 rounded text-xs hover:bg-purple-700"
            >
              View Complete
            </button>
          </div>
        </div>
      );
    
    case 'editor':
      return (
        <div>
          <h4 className="font-medium mb-2">Review Completed:</h4>
          <div className="space-y-2">
            <p className="text-sm text-gray-600">
              Score: {data.editor_feedback?.final_score || data.editor_feedback?.score || 'N/A'}/100
            </p>
            {data.editor_feedback?.suggestions && (
              <div>
                <p className="text-xs font-medium text-gray-700 mb-1">Suggestions:</p>
                <ul className="text-xs text-gray-600 space-y-1">
                  {data.editor_feedback.suggestions.slice(0, 3).map((suggestion: string, index: number) => (
                    <li key={index} className="flex items-start">
                      <span className="text-blue-500 mr-1">•</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      );
    
    case 'image':
      return (
        <div>
          <h4 className="font-medium mb-2">Generated Images:</h4>
          <p className="text-sm text-gray-600">
            {data.images?.length || 0} images generated
          </p>
          {data.images && data.images.length > 0 && (
            <div className="mt-2 space-y-1">
              {data.images.slice(0, 2).map((img: any, index: number) => (
                <div key={index} className="text-xs text-gray-600">
                  • {img.alt_text || `Image ${index + 1}`}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    
    case 'seo':
      return (
        <div>
          <h4 className="font-medium mb-2">SEO Analysis:</h4>
          <p className="text-sm text-gray-600">
            Score: {data.seo_analysis?.seo_score || 'N/A'}/100
          </p>
          {data.seo_analysis?.recommendations && (
            <div className="mt-2">
              <p className="text-xs font-medium text-gray-700 mb-1">Recommendations:</p>
              <ul className="text-xs text-gray-600 space-y-1">
                {data.seo_analysis.recommendations.slice(0, 2).map((rec: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="text-green-500 mr-1">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    
    case 'social_media':
      return (
        <div>
          <h4 className="font-medium mb-2">Social Media Posts:</h4>
          <p className="text-sm text-gray-600">
            {Object.keys(data.social_posts || {}).length} platforms processed
          </p>
          {data.social_posts && (
            <div className="mt-2 space-y-1">
              {Object.keys(data.social_posts).slice(0, 3).map((platform: string) => (
                <div key={platform} className="text-xs text-gray-600 capitalize">
                  • {platform}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    
    default:
      return null;
  }
};

export default WorkflowWizard; 