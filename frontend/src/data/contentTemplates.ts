export interface ContentTemplate {
  id: string;
  name: string;
  industry: string;
  description: string;
  icon: string;
  titleSuggestions: string[];
  companyContextTemplate: string;
  contentOutline: string[];
  targetAudience: string;
  toneOfVoice: string;
  keyPoints: string[];
  callToAction: string[];
}

export const contentTemplates: ContentTemplate[] = [
  {
    id: 'saas-thought-leadership',
    name: 'SaaS Thought Leadership',
    industry: 'SaaS & Technology',
    description: 'Establish authority in the SaaS space with industry insights and trends',
    icon: '💡',
    titleSuggestions: [
      'The Future of [Your Technology]: What Industry Leaders Need to Know',
      '5 SaaS Trends That Will Transform Business Operations in 2024',
      'From Startup to Scale: Lessons Learned Building a [Your Category] Platform',
      'Why Traditional [Industry] is Ready for Digital Transformation'
    ],
    companyContextTemplate: `[Company Name] is a leading SaaS platform specializing in [specific solution/category]. We serve [target market segment] by providing [key value proposition]. Our platform helps businesses [main benefits/outcomes]. We're known for our expertise in [core competencies] and have helped over [number] companies achieve [specific results].`,
    contentOutline: [
      'Industry problem identification and market context',
      'Current state analysis with data and statistics',
      'Emerging trends and future predictions',
      'Practical implementation strategies',
      'Case studies and success stories',
      'Actionable next steps for readers'
    ],
    targetAudience: 'C-level executives, IT decision makers, and business operations managers',
    toneOfVoice: 'Authoritative yet approachable, data-driven, forward-thinking',
    keyPoints: [
      'Demonstrate deep industry knowledge',
      'Provide actionable insights backed by data',
      'Address common pain points and challenges',
      'Position your solution as the natural choice'
    ],
    callToAction: [
      'Schedule a demo to see how our platform addresses these challenges',
      'Download our comprehensive industry report',
      'Join our upcoming webinar on [related topic]',
      'Contact our experts for a personalized consultation'
    ]
  },
  {
    id: 'ecommerce-growth',
    name: 'E-commerce Growth Guide',
    industry: 'E-commerce & Retail',
    description: 'Drive sales and engagement with actionable e-commerce strategies',
    icon: '🛒',
    titleSuggestions: [
      'How to Increase E-commerce Conversion Rates by 30% in 90 Days',
      'The Complete Guide to [Season] E-commerce Marketing',
      '7 E-commerce Trends Every Online Retailer Must Know',
      'From Cart Abandonment to Completed Sale: A Data-Driven Approach'
    ],
    companyContextTemplate: `[Company Name] is an innovative e-commerce [platform/solution/service] that helps online retailers [primary function]. We specialize in [key differentiators] and have enabled merchants to achieve [average results/improvements]. Our clients range from [customer segments] who trust us to [main value delivered].`,
    contentOutline: [
      'Current e-commerce landscape and challenges',
      'Key performance metrics and benchmarks',
      'Step-by-step strategy implementation',
      'Tools and technologies needed',
      'Real-world case studies with results',
      'Common pitfalls and how to avoid them'
    ],
    targetAudience: 'E-commerce managers, digital marketing specialists, and online business owners',
    toneOfVoice: 'Practical, results-focused, encouraging, with clear ROI emphasis',
    keyPoints: [
      'Focus on measurable outcomes and ROI',
      'Provide step-by-step implementation guides',
      'Include relevant statistics and benchmarks',
      'Address seasonal and trending topics'
    ],
    callToAction: [
      'Start your free trial to implement these strategies',
      'Get a personalized e-commerce audit',
      'Download our conversion optimization toolkit',
      'Book a strategy call with our e-commerce experts'
    ]
  },
  {
    id: 'fintech-innovation',
    name: 'FinTech Innovation',
    industry: 'Financial Technology',
    description: 'Explore financial innovation and regulatory compliance insights',
    icon: '💳',
    titleSuggestions: [
      'The Evolution of Digital Payments: What\'s Next for FinTech',
      'Regulatory Compliance in FinTech: A Complete Guide',
      'How Embedded Finance is Transforming Traditional Banking',
      'Building Trust in Digital Financial Services: Security Best Practices'
    ],
    companyContextTemplate: `[Company Name] is a cutting-edge FinTech company providing [specific financial services/solutions]. We operate in [geographic markets] and specialize in [core competencies like embedded lending, digital payments, etc.]. Our platform serves [target customers] by offering [key benefits] while maintaining the highest standards of [security/compliance/innovation].`,
    contentOutline: [
      'Current financial services landscape',
      'Regulatory environment and compliance requirements',
      'Technology trends and innovations',
      'Implementation challenges and solutions',
      'Security and risk management considerations',
      'Future outlook and strategic recommendations'
    ],
    targetAudience: 'Financial services executives, compliance officers, and technology decision makers',
    toneOfVoice: 'Professional, trustworthy, technically accurate, compliance-aware',
    keyPoints: [
      'Emphasize security and regulatory compliance',
      'Provide technical depth with business context',
      'Address risk management and mitigation',
      'Showcase innovation while maintaining trust'
    ],
    callToAction: [
      'Learn more about our compliant financial solutions',
      'Request a security and compliance audit',
      'Schedule a consultation with our FinTech experts',
      'Download our regulatory compliance checklist'
    ]
  },
  {
    id: 'healthcare-digital',
    name: 'Healthcare Digital Transformation',
    industry: 'Healthcare & MedTech',
    description: 'Navigate healthcare technology adoption and patient care improvements',
    icon: '🏥',
    titleSuggestions: [
      'Digital Health Revolution: Improving Patient Outcomes Through Technology',
      'HIPAA Compliance in the Age of Digital Healthcare Solutions',
      'Telemedicine Best Practices: Lessons from the Pandemic',
      'AI in Healthcare: Promise, Challenges, and Practical Applications'
    ],
    companyContextTemplate: `[Company Name] is a healthcare technology company focused on [specific healthcare area]. We develop [type of solutions] that help [healthcare providers/patients/administrators] achieve [key outcomes]. Our platform is designed with [key features like HIPAA compliance, interoperability, etc.] and has been adopted by [types of healthcare organizations] to improve [specific metrics].`,
    contentOutline: [
      'Current healthcare challenges and opportunities',
      'Technology solutions and their impact',
      'Regulatory compliance and privacy considerations',
      'Implementation strategies and best practices',
      'Patient outcomes and provider benefits',
      'Future trends and recommendations'
    ],
    targetAudience: 'Healthcare administrators, medical professionals, and health IT decision makers',
    toneOfVoice: 'Professional, empathetic, evidence-based, patient-focused',
    keyPoints: [
      'Prioritize patient outcomes and safety',
      'Address regulatory and privacy requirements',
      'Provide evidence-based recommendations',
      'Consider both provider and patient perspectives'
    ],
    callToAction: [
      'Schedule a demo of our healthcare solution',
      'Download our healthcare compliance guide',
      'Join our healthcare innovation webinar',
      'Contact our medical advisory team'
    ]
  },
  {
    id: 'education-tech',
    name: 'EdTech Innovation',
    industry: 'Education Technology',
    description: 'Transform learning experiences with educational technology insights',
    icon: '📚',
    titleSuggestions: [
      'The Future of Learning: How EdTech is Reshaping Education',
      'Personalized Learning at Scale: Technology Solutions That Work',
      'Measuring Learning Outcomes in Digital Education Environments',
      'Bridging the Digital Divide: Making EdTech Accessible to All'
    ],
    companyContextTemplate: `[Company Name] is an innovative EdTech company dedicated to [educational mission]. We provide [type of educational solutions] for [target learners/institutions]. Our platform focuses on [key educational outcomes] and has helped [number] students/educators/institutions achieve [specific results]. We specialize in [core competencies like personalized learning, assessment, etc.].`,
    contentOutline: [
      'Current state of education and technology adoption',
      'Learning challenges and technology solutions',
      'Implementation strategies for educational institutions',
      'Measuring success and learning outcomes',
      'Accessibility and equity considerations',
      'Future of education technology'
    ],
    targetAudience: 'Educators, administrators, curriculum developers, and educational decision makers',
    toneOfVoice: 'Inspiring, evidence-based, inclusive, focused on learning outcomes',
    keyPoints: [
      'Center content around student success',
      'Provide practical implementation guidance',
      'Address accessibility and equity issues',
      'Include pedagogical considerations'
    ],
    callToAction: [
      'Try our educational platform free for 30 days',
      'Download our learning outcomes research report',
      'Join our educator community forum',
      'Schedule a consultation with our education experts'
    ]
  },
  {
    id: 'manufacturing-industry40',
    name: 'Industry 4.0 Manufacturing',
    industry: 'Manufacturing & Industrial',
    description: 'Drive operational efficiency through smart manufacturing technologies',
    icon: '🏭',
    titleSuggestions: [
      'Industry 4.0: Transforming Manufacturing with Smart Technology',
      'Predictive Maintenance: Reducing Downtime and Costs',
      'Digital Twin Technology: Optimizing Production Processes',
      'Supply Chain Resilience in the Age of Smart Manufacturing'
    ],
    companyContextTemplate: `[Company Name] is a leading provider of [manufacturing solutions/technologies]. We help manufacturers in [specific industries] optimize their operations through [key technologies/solutions]. Our platform specializes in [core capabilities] and has enabled clients to achieve [typical improvements in efficiency, cost reduction, etc.]. We serve [customer segments] across [geographic regions].`,
    contentOutline: [
      'Current manufacturing challenges and opportunities',
      'Industry 4.0 technologies and applications',
      'Implementation roadmap and best practices',
      'ROI and performance metrics',
      'Case studies from successful deployments',
      'Future trends and strategic planning'
    ],
    targetAudience: 'Manufacturing executives, operations managers, and industrial engineers',
    toneOfVoice: 'Technical, practical, ROI-focused, safety-conscious',
    keyPoints: [
      'Emphasize operational efficiency and cost savings',
      'Provide technical depth with practical applications',
      'Address safety and quality considerations',
      'Include measurable business outcomes'
    ],
    callToAction: [
      'Schedule a facility assessment with our engineers',
      'Download our Industry 4.0 implementation guide',
      'Request a ROI analysis for your operations',
      'Join our manufacturing innovation summit'
    ]
  }
];

export const getTemplateById = (id: string): ContentTemplate | undefined => {
  return contentTemplates.find(template => template.id === id);
};

export const getTemplatesByIndustry = (industry: string): ContentTemplate[] => {
  return contentTemplates.filter(template => template.industry === industry);
};

export const getAllIndustries = (): string[] => {
  return [...new Set(contentTemplates.map(template => template.industry))];
};