import React, { useState, useEffect } from 'react';
import { ChevronRightIcon, CheckIcon } from '@heroicons/react/24/outline';

interface WorkflowStep {
  id: 'planner' | 'researcher' | 'writer' | 'editor';
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
  blog_title: string;
  company_context: string;
}

const WorkflowWizard: React.FC = () => {
  const [workflowState, setWorkflowState] = useState<WorkflowState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const steps: WorkflowStep[] = [
    {
      id: 'planner',
      title: 'Planificación',
      description: 'Crear outline estructurado del contenido',
      status: 'pending'
    },
    {
      id: 'researcher',
      title: 'Investigación',
      description: 'Investigar cada sección del outline',
      status: 'pending'
    },
    {
      id: 'writer',
      title: 'Escritura',
      description: 'Generar contenido basado en la investigación',
      status: 'pending'
    },
    {
      id: 'editor',
      title: 'Revisión',
      description: 'Revisar y aprobar el contenido final',
      status: 'pending'
    }
  ];

  const startWorkflow = async (title: string, context: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/workflow/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, company_context: context })
      });
      
      if (!response.ok) throw new Error('Error al iniciar el workflow');
      
      const data = await response.json();
      setWorkflowState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsLoading(false);
    }
  };

  const executeStep = async (stepId: WorkflowStep['id']) => {
    if (!workflowState) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/workflow/${stepId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow_id: workflowState.workflow_id })
      });
      
      if (!response.ok) throw new Error(`Error en el paso ${stepId}`);
      
      const data = await response.json();
      setWorkflowState(prev => ({ ...prev, ...data }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setIsLoading(false);
    }
  };

  const getStepStatus = (stepId: WorkflowStep['id']): WorkflowStep['status'] => {
    if (!workflowState) return 'pending';
    
    if (workflowState.current_step === stepId) return 'in-progress';
    if (workflowState.progress > steps.findIndex(s => s.id === stepId)) return 'completed';
    return 'pending';
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Generador de Contenido Inteligente
        </h1>
        <p className="text-gray-600">
          Crea contenido profesional usando agentes especializados
        </p>
      </div>

      {/* Input inicial */}
      {!workflowState && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Iniciar Nuevo Contenido</h2>
          <WorkflowInput onStart={startWorkflow} isLoading={isLoading} />
        </div>
      )}

      {/* Progreso del workflow */}
      {workflowState && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4">Progreso del Workflow</h2>
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

          {/* Pasos del workflow */}
          <div className="space-y-4">
            {steps.map((step, index) => (
              <WorkflowStep
                key={step.id}
                step={step}
                status={getStepStatus(step.id)}
                isActive={workflowState.current_step === step.id}
                onExecute={() => executeStep(step.id)}
                isLoading={isLoading && workflowState.current_step === step.id}
                data={workflowState}
              />
            ))}
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
          Título del Contenido
        </label>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Ej: Guía completa de marketing digital"
          required
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Contexto de la Empresa
        </label>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          placeholder="Describe tu empresa, público objetivo, tono de comunicación..."
          required
        />
      </div>
      
      <button
        type="submit"
        disabled={isLoading || !title.trim() || !context.trim()}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? 'Iniciando...' : 'Comenzar Workflow'}
      </button>
    </form>
  );
};

// Componente para cada paso del workflow
const WorkflowStep: React.FC<{
  step: WorkflowStep;
  status: WorkflowStep['status'];
  isActive: boolean;
  onExecute: () => void;
  isLoading: boolean;
  data: WorkflowState;
}> = ({ step, status, isActive, onExecute, isLoading, data }) => {
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
        
        {isActive && (
          <button
            onClick={onExecute}
            disabled={isLoading}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {isLoading ? 'Ejecutando...' : 'Ejecutar'}
          </button>
        )}
      </div>

      {/* Mostrar datos del paso si están disponibles */}
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
          <h4 className="font-medium mb-2">Outline Generado:</h4>
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
          <h4 className="font-medium mb-2">Investigación Completada:</h4>
          <p className="text-sm text-gray-600">
            {Object.keys(data.research || {}).length} secciones investigadas
          </p>
        </div>
      );
    
    case 'writer':
      return (
        <div>
          <h4 className="font-medium mb-2">Contenido Generado:</h4>
          <div className="text-sm text-gray-700 max-h-32 overflow-y-auto">
            {data.content?.substring(0, 200)}...
          </div>
        </div>
      );
    
    case 'editor':
      return (
        <div>
          <h4 className="font-medium mb-2">Revisión Completada:</h4>
          <p className="text-sm text-gray-600">
            Puntuación: {data.editor_feedback?.score || 'N/A'}/100
          </p>
        </div>
      );
    
    default:
      return null;
  }
};

export default WorkflowWizard; 