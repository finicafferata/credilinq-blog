import React from 'react';
import WorkflowWizard from '../components/WorkflowWizard';
import { Link } from 'react-router-dom';

const WorkflowPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-2 text-right">
        <Link to="/settings" className="text-sm text-blue-600">Edit default company context</Link>
      </div>
      <WorkflowWizard />
    </div>
  );
};

export default WorkflowPage; 