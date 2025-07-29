import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { blogApi } from '../lib/api';
import { showErrorNotification, AppError } from '../lib/errors';
import { Breadcrumbs } from '../components/Breadcrumbs';
import { KeyboardShortcutsHelp } from '../components/KeyboardShortcutsHelp';
import { ProgressBar } from '../components/LoadingStates';
import { contentTemplates, type ContentTemplate } from '../data/contentTemplates';

interface BlogFormData {
  title: string;
  company_context: string;
  content_type: 'blog' | 'linkedin';
  template?: ContentTemplate;
  industry?: string;
  targetAudience?: string;
  toneOfVoice?: string;
  keyPoints?: string[];
  callToAction?: string;
}

interface StepProps {
  formData: BlogFormData;
  setFormData: (data: BlogFormData) => void;
  onNext: () => void;
  onBack: () => void;
  isValid: boolean;
}

// Step 1: Content Type Selection
function Step1ContentType({ formData, setFormData, onNext, isValid }: StepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Content Type</h2>
        <p className="text-gray-600">Select the type of content you want to create</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <button
          type="button"
          onClick={() => setFormData({ ...formData, content_type: 'blog' })}
          className={`p-6 rounded-xl border-2 text-left transition-all hover:shadow-lg ${
            formData.content_type === 'blog'
              ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
              : 'border-gray-200 bg-white hover:border-gray-300'
          }`}
        >
          <div className="flex items-center mb-4">
            <div className="text-3xl mr-4">📝</div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Blog Post</h3>
              <p className="text-sm text-gray-600">Comprehensive articles</p>
            </div>
          </div>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• 1500-3000 words</li>
            <li>• SEO optimized</li>
            <li>• In-depth coverage</li>
            <li>• Multiple sections</li>
          </ul>
        </button>

        <button
          type="button"
          onClick={() => setFormData({ ...formData, content_type: 'linkedin' })}
          className={`p-6 rounded-xl border-2 text-left transition-all hover:shadow-lg ${
            formData.content_type === 'linkedin'
              ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
              : 'border-gray-200 bg-white hover:border-gray-300'
          }`}
        >
          <div className="flex items-center mb-4">
            <div className="text-3xl mr-4">💼</div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">LinkedIn Post</h3>
              <p className="text-sm text-gray-600">Professional content</p>
            </div>
          </div>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• 800-1500 words</li>
            <li>• Engagement focused</li>
            <li>• Professional tone</li>
            <li>• Social sharing ready</li>
          </ul>
        </button>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onNext}
          disabled={!isValid}
          className="btn-primary disabled:opacity-50"
        >
          Continue
        </button>
      </div>
    </div>
  );
}

// Step 2: Industry Template Selection
function Step2Template({ formData, setFormData, onNext, onBack }: StepProps) {
  const handleTemplateSelect = (template: ContentTemplate) => {
    setFormData({
      ...formData,
      template,
      industry: template.industry,
      targetAudience: template.targetAudience,
      toneOfVoice: template.toneOfVoice,
      keyPoints: template.keyPoints,
      callToAction: template.callToAction[0]
    });
  };

  const handleSkipTemplate = () => {
    setFormData({
      ...formData,
      template: undefined,
      industry: undefined,
      targetAudience: undefined,
      toneOfVoice: undefined,
      keyPoints: undefined,
      callToAction: undefined
    });
    onNext();
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose an Industry Template</h2>
        <p className="text-gray-600">
          Select a template tailored to your industry or skip to create custom content
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {contentTemplates.map((template) => (
          <button
            key={template.id}
            type="button"
            onClick={() => handleTemplateSelect(template)}
            className={`p-4 rounded-lg border-2 text-left transition-all hover:shadow-lg ${
              formData.template?.id === template.id
                ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-200'
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <div className="flex items-start mb-3">
              <span className="text-2xl mr-3">{template.icon}</span>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 text-sm">{template.name}</h3>
                <p className="text-xs text-gray-600 mt-1">{template.industry}</p>
              </div>
            </div>
            <p className="text-xs text-gray-700 leading-relaxed">{template.description}</p>
          </button>
        ))}
      </div>

      {formData.template && (
        <div className="bg-blue-50 rounded-lg p-4 mt-6">
          <h4 className="font-semibold text-blue-900 mb-2">Selected Template: {formData.template.name}</h4>
          <p className="text-sm text-blue-800 mb-3">{formData.template.description}</p>
          <div className="text-xs text-blue-700">
            <p><strong>Target Audience:</strong> {formData.template.targetAudience}</p>
            <p><strong>Tone:</strong> {formData.template.toneOfVoice}</p>
          </div>
        </div>
      )}

      <div className="flex justify-between">
        <button onClick={onBack} className="btn-secondary">
          Back
        </button>
        <div className="space-x-3">
          <button onClick={handleSkipTemplate} className="btn-secondary">
            Skip Template
          </button>
          <button
            onClick={onNext}
            disabled={!formData.template}
            className="btn-primary disabled:opacity-50"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}

// Step 3: Content Details
function Step3Details({ formData, setFormData, onNext, onBack, isValid }: StepProps) {
  const suggestedTitles = formData.template?.titleSuggestions || [];

  const handleTitleSuggestion = (title: string) => {
    setFormData({ ...formData, title });
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Content Details</h2>
        <p className="text-gray-600">Provide the key information for your content</p>
      </div>

      <div className="space-y-6">
        <div>
          <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
            {formData.content_type === 'linkedin' ? 'Post Title' : 'Blog Title'} *
          </label>
          <input
            type="text"
            id="title"
            className="input"
            placeholder="Enter your content title..."
            value={formData.title}
            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
          />
          
          {suggestedTitles.length > 0 && (
            <div className="mt-3">
              <p className="text-sm font-medium text-gray-700 mb-2">Suggested titles:</p>
              <div className="space-y-2">
                {suggestedTitles.map((title, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => handleTitleSuggestion(title)}
                    className="block w-full text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors text-sm"
                  >
                    {title}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div>
          <label htmlFor="context" className="block text-sm font-medium text-gray-700 mb-2">
            Company Context *
          </label>
          <textarea
            id="context"
            rows={6}
            className="textarea"
            placeholder={formData.template?.companyContextTemplate || "Describe your company, products, and target audience..."}
            value={formData.company_context}
            onChange={(e) => setFormData({ ...formData, company_context: e.target.value })}
          />
          <p className="mt-2 text-sm text-gray-500">
            This context helps our AI understand your company's voice and focus areas.
          </p>
        </div>

        {formData.template && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Content Outline Preview</h4>
            <ol className="text-sm text-gray-700 space-y-1">
              {formData.template.contentOutline.map((item, index) => (
                <li key={index} className="flex items-start">
                  <span className="font-medium text-blue-600 mr-2">{index + 1}.</span>
                  <span className="capitalize">{item}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button onClick={onBack} className="btn-secondary">
          Back
        </button>
        <button
          onClick={onNext}
          disabled={!isValid}
          className="btn-primary disabled:opacity-50"
        >
          Continue
        </button>
      </div>
    </div>
  );
}

// Step 4: Review and Create
function Step4Review({ formData, onBack }: StepProps & { onSubmit: () => void; isLoading: boolean }) {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    try {
      setIsLoading(true);
      const newBlog = await blogApi.create({
        title: formData.title,
        company_context: formData.company_context,
        content_type: formData.content_type
      });
      navigate(`/edit/${newBlog.id}`);
    } catch (error) {
      showErrorNotification(error instanceof AppError ? error : new AppError('Failed to create blog. Please try again.'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Review & Create</h2>
        <p className="text-gray-600">Review your content settings before creation</p>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Content Summary</h3>
        </div>
        
        <div className="p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <dt className="text-sm font-medium text-gray-500">Content Type</dt>
              <dd className="mt-1 text-sm text-gray-900 capitalize">{formData.content_type}</dd>
            </div>
            
            {formData.template && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Template</dt>
                <dd className="mt-1 text-sm text-gray-900">{formData.template.name}</dd>
              </div>
            )}
            
            {formData.industry && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Industry</dt>
                <dd className="mt-1 text-sm text-gray-900">{formData.industry}</dd>
              </div>
            )}
            
            {formData.targetAudience && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Target Audience</dt>
                <dd className="mt-1 text-sm text-gray-900">{formData.targetAudience}</dd>
              </div>
            )}
          </div>
          
          <div>
            <dt className="text-sm font-medium text-gray-500">Title</dt>
            <dd className="mt-1 text-sm text-gray-900 font-medium">{formData.title}</dd>
          </div>
          
          <div>
            <dt className="text-sm font-medium text-gray-500">Company Context</dt>
            <dd className="mt-1 text-sm text-gray-900">{formData.company_context}</dd>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">What happens next:</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Our AI agents will analyze your requirements</li>
          <li>• Content will be generated based on your template and context</li>
          <li>• You'll be redirected to the editor for final customization</li>
          <li>• Real-time preview and collaboration features will be available</li>
        </ul>
      </div>

      <div className="flex justify-between">
        <button onClick={onBack} className="btn-secondary" disabled={isLoading}>
          Back
        </button>
        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="btn-primary disabled:opacity-50"
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Creating...</span>
            </div>
          ) : (
            `Create ${formData.content_type === 'linkedin' ? 'LinkedIn Post' : 'Blog Post'}`
          )}
        </button>
      </div>
    </div>
  );
}

export function EnhancedBlogCreator() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<BlogFormData>({
    title: '',
    company_context: 'Credilinq.ai is a global fintech leader in embedded lending and B2B credit solutions, operating across Southeast Asia, Europe, and the United States. We empower businesses to access funding through embedded financial products and cutting-edge credit infrastructure tailored to digital platforms and marketplaces.',
    content_type: 'blog'
  });

  const totalSteps = 4;
  const progress = (currentStep / totalSteps) * 100;

  const nextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 1:
        return !!formData.content_type;
      case 2:
        return true; // Template is optional
      case 3:
        return !!(formData.title.trim() && formData.company_context.trim());
      case 4:
        return true;
      default:
        return false;
    }
  };

  const renderCurrentStep = () => {
    const stepProps = {
      formData,
      setFormData,
      onNext: nextStep,
      onBack: prevStep,
      isValid: isStepValid(currentStep)
    };

    switch (currentStep) {
      case 1:
        return <Step1ContentType {...stepProps} />;
      case 2:
        return <Step2Template {...stepProps} />;
      case 3:
        return <Step3Details {...stepProps} />;
      case 4:
        return <Step4Review {...stepProps} onSubmit={() => {}} isLoading={false} />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <Breadcrumbs />
        
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Create New Content</h1>
            <p className="text-gray-600">
              Follow our guided process to create optimized content with AI assistance
            </p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Step {currentStep} of {totalSteps}</span>
              <span className="text-sm text-gray-500">{Math.round(progress)}% complete</span>
            </div>
            <ProgressBar progress={progress} size="md" />
          </div>

          {/* Step Navigation */}
          <div className="flex items-center justify-center mb-8">
            <div className="flex items-center space-x-4">
              {Array.from({ length: totalSteps }, (_, i) => i + 1).map((step) => (
                <div key={step} className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
                      step === currentStep
                        ? 'bg-primary-600 text-white ring-4 ring-primary-100'
                        : step < currentStep
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-200 text-gray-600'
                    }`}
                  >
                    {step < currentStep ? (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      step
                    )}
                  </div>
                  {step < totalSteps && (
                    <div
                      className={`w-12 h-0.5 mx-2 transition-all ${
                        step < currentStep ? 'bg-green-500' : 'bg-gray-200'
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Step Content */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
            {renderCurrentStep()}
          </div>

          {/* Help Text */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-500">
              Need help? Press{' '}
              <kbd className="px-2 py-1 bg-gray-200 rounded text-xs font-mono">?</kbd>{' '}
              for keyboard shortcuts or{' '}
              <button
                onClick={() => navigate('/dashboard')}
                className="text-primary-600 hover:text-primary-700 underline"
              >
                return to dashboard
              </button>
            </p>
          </div>
        </div>

        <KeyboardShortcutsHelp />
      </div>
    </div>
  );
}