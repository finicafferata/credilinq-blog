import { useBlogAnalytics } from '../hooks/useAnalytics';

interface BlogAnalyticsWidgetProps {
  blogId: string;
}

export function BlogAnalyticsWidget({ blogId }: BlogAnalyticsWidgetProps) {
  const { analytics, isLoading, error } = useBlogAnalytics(blogId);

  if (isLoading) {
    return (
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-2"></div>
          <div className="h-6 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-600 text-sm">{error}</div>
      </div>
    );
  }

  if (!analytics) {
    return null;
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h4 className="text-sm font-medium text-gray-900 mb-3">Blog Analytics</h4>
      
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">{analytics.views}</div>
          <div className="text-xs text-gray-500">Views</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900">{analytics.unique_visitors}</div>
          <div className="text-xs text-gray-500">Unique Visitors</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {(analytics.engagement_rate * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Engagement</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{analytics.social_shares}</div>
          <div className="text-xs text-gray-500">Shares</div>
        </div>
      </div>
      
      {analytics.seo_score > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">SEO Score</span>
            <span className="text-sm font-medium text-gray-900">
              {(analytics.seo_score * 100).toFixed(0)}/100
            </span>
          </div>
          <div className="mt-1 h-2 bg-gray-200 rounded-full">
            <div 
              className="h-2 bg-green-500 rounded-full" 
              style={{ width: `${analytics.seo_score * 100}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
}