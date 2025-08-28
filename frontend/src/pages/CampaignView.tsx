import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useCampaignPlan } from '../hooks/useCampaignPlan';
import { CampaignDetails } from '../components/CampaignDetails';
import { RepurposeModal } from '../components/RepurposeModal';
import type { CampaignDetail } from '../lib/api';

// CampaignView is the main page for managing a campaign workflow for a blog post
function CampaignView() {
  // Get the blogId or campaignId from the route params
  const { blogId, campaignId } = useParams<{ blogId?: string; campaignId?: string }>();
  
  // Use campaignId if available, otherwise use a mock ID
  const actualCampaignId = campaignId || "test-campaign-1";
  
  // Modal states
  const [showRepurpose, setShowRepurpose] = useState(false);
  
  // Use the custom hook to manage campaign state
  const {
    campaign,
    isLoading,
    error,
    fetchCampaign,
    schedule,
    distribute,
    updateTaskStatus,
  } = useCampaignPlan(actualCampaignId);

  // Fetch the campaign on mount
  useEffect(() => {
    fetchCampaign();
  }, [fetchCampaign]);

  // Handler for repurposed content
  const handleRepurpose = (repurposedContent: any) => {
    console.log('Repurposed content:', repurposedContent);
    // Here you would typically save the repurposed content to the campaign
    // For now, we'll just log it
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-600">Loading campaign...</p>
        </div>
      </div>
    );
  }

  if (error || !campaign) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-red-800 mb-2">Error Loading Campaign</h2>
            <p className="text-red-600">{error || 'Campaign not found'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Campaign Details - Full Page */}
      <CampaignDetails
        campaign={campaign}
        onClose={() => window.history.back()}
        fullPage={true}
      />

      {/* Repurpose Modal */}
      {showRepurpose && (
        <RepurposeModal
          isOpen={showRepurpose}
          onClose={() => setShowRepurpose(false)}
          onRepurpose={handleRepurpose}
          blogContent=""
        />
      )}
    </div>
  );
}

export default CampaignView;