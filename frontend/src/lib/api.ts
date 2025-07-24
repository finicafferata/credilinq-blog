import axios from 'axios';

const isDev = import.meta.env.DEV;

const api = axios.create({
  baseURL: isDev ? 'http://localhost:8000' : '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface BlogSummary {
  id: string;
  title: string;
  status: string;
  created_at: string;
}

export interface BlogDetail extends BlogSummary {
  content_markdown: string;
  initial_prompt: unknown;
}

export interface BlogCreateRequest {
  title: string;
  company_context: string;
  content_type: string; // "blog" or "linkedin"
}

export interface BlogEditRequest {
  content_markdown: string;
}

export interface BlogReviseRequest {
  instruction: string;
  text_to_revise: string;
}

// Blog API endpoints
export const blogApi = {
  // Create a new blog post
  create: async (data: BlogCreateRequest): Promise<BlogSummary> => {
    const response = await api.post('/blogs', data);
    return response.data;
  },

  // Get all blogs
  list: async (): Promise<BlogSummary[]> => {
    const response = await api.get('/blogs');
    return response.data;
  },

  // Get single blog by ID
  get: async (id: string): Promise<BlogDetail> => {
    const response = await api.get(`/blogs/${id}`);
    return response.data;
  },

  // Update blog content
  update: async (id: string, data: BlogEditRequest): Promise<BlogDetail> => {
    const response = await api.put(`/blogs/${id}`, data);
    return response.data;
  },

  // AI revision of text snippet
  revise: async (id: string, data: BlogReviseRequest): Promise<{ revised_text: string }> => {
    const response = await api.post(`/blogs/${id}/revise`, data);
    return response.data;
  },
};

// Documents API endpoints
export const documentsApi = {
  // Upload document for RAG
  upload: async (file: File, title?: string): Promise<{ document_id: string; storage_path: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) {
      formData.append('document_title', title);
    }

    const response = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default api; 