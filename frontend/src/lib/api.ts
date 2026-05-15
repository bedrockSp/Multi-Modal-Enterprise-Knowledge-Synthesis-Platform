import { API_URL } from '../../config';

// Types
export interface User {
  userId: string;
  name: string;
  email: string;
  threads: Record<string, Thread>;
}

export interface ThreadInstruction {
  id: string;
  text: string;
  selected: boolean;
}

export interface Thread {
  thread_name: string;
  createdAt: string;
  updatedAt: string;
  documents: Document[];
  chats: Chat[];
  instructions?: ThreadInstruction[];
}

export interface Document {
  docId: string;
  title: string;
  type: string;
  time_uploaded: string;
  file_name: string;
}

export interface Chat {
  type: 'user' | 'agent';
  content: string;
  timestamp: string;
  // Enhanced metadata fields from Phase 1/2 backend
  confidence_score?: string | number; // Backend sends "high"/"medium"/"low"
  thought_process?: string; // For Deep Reasoning output
  sources?: {
    documents_used: Array<{
      title: string;
      document_id: string;
      page_no: number;
    }>;
    web_used: Array<{
      title: string;
      url: string;
      favicon: string | null;
    }>;
  };
}

export interface DeleteChatResponse {
  status: string;
  message: string;
  thread_id: string;
  deleted_index: number;
  chats: Chat[];
}

export interface ClearChatsResponse {
  status: string;
  message: string;
  thread_id: string;
  chats: Chat[];
}

export interface DeleteDocumentResponse {
  status: string;
  message: string;
  thread_id: string;
  deleted_doc_id: string;
  documents: Document[];
}

export interface AddExistingDocumentResponse {
  status: string;
  message: string;
  thread_id: string;
  document: Document;
}

export interface LoginResponse {
  status: string;
  message: string;
  user: User;
  token: string;
}

export interface UploadResponse {
  status: string;
  message: string;
  thread_id: string;
  documents: Document[];
}

export interface QueryResponse {
  thread_id: string;
  user_id: string;
  question: string;
  answer: string;
  use_self_knowledge?: boolean;
  // Enhanced metadata fields
  confidence_score?: string | number;
  thought_process?: string;
  // Original shape (legacy)
  docs_used?: Array<{
    title: string;
    document_id: string;
    page_no: number;
  }>;
  web_used?: Array<{
    title: string;
    url: string;
    favicon: string | null;
  }>;
  // Newer shape returned by backend under a `sources` object
  sources?: {
    documents_used?: Array<{
      title: string;
      document_id: string;
      page_no: number;
    }>;
    web_used?: Array<{
      title: string;
      url: string;
      favicon: string | null;
    }>;
  };
}

// Mind map types
export interface MindMapNode {
  id: string;
  title: string;
  description?: string | null;
  parent_id?: string | null;
  children: MindMapNode[];
}

export interface GlobalMindMap {
  user_id: string;
  thread_id: string;
  roots: MindMapNode[];
}

export interface MindMapResponse {
  mind_map: boolean;
  status?: boolean; // only present when mind_map is true
  message: string;
  data?: GlobalMindMap; // present when mind_map && status
}

export interface SummaryResponse {
  status?: boolean;
  summary?: string;
  message?: string;
  error?: string;
  failed?: boolean;
}

// Roadmap types (mirror backend Pydantic models)
export interface VisionAndEndGoal {
  description: string;
  success_criteria: string[];
}

export interface SWOT {
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
}

export interface CurrentBaseline {
  summary: string;
  swot: SWOT;
}

export interface StrategicPillar {
  pillar_name: string;
  description: string;
}

export interface PhasedRoadmapItem {
  phase: string;
  time_frame: string;
  key_objectives: string[];
  key_initiatives: string[];
  expected_outcomes: string[];
}

export interface EnablersAndDependencies {
  technologies: string[];
  skills_and_resources: string[];
  stakeholders: string[];
}

export interface RiskAndMitigation {
  risk: string;
  mitigation_strategy: string;
}

export interface KeyMetricsAndMilestone {
  year_or_phase: string;
  metrics: string[];
}

export interface LLMInferredAddition {
  section_title: string;
  content: string;
}

export interface StrategicRoadmapLLMOutput {
  roadmap_title: string;
  vision_and_end_goal: VisionAndEndGoal;
  current_baseline: CurrentBaseline;
  strategic_pillars: StrategicPillar[];
  phased_roadmap: PhasedRoadmapItem[];
  enablers_and_dependencies: EnablersAndDependencies;
  risks_and_mitigation: RiskAndMitigation[];
  key_metrics_and_milestones: KeyMetricsAndMilestone[];
  future_opportunities: string[];
  llm_inferred_additions: LLMInferredAddition[];
}

export interface StrategicRoadmapResponse {
  status?: boolean;
  strategic_roadmap?: StrategicRoadmapLLMOutput;
  message?: string;
  error?: string;
  failed?: boolean;
}

// Technical Roadmap types (mirror backend Pydantic models provided)
export interface OverallVision {
  goal: string;
  success_metrics: string[];
}

export interface CurrentStateAnalysis {
  summary: string;
  key_challenges: string[];
  existing_capabilities: string[];
}

export interface TechnologyDomain {
  domain_name: string;
  description: string;
}

export interface Initiative {
  initiative: string;
  objective: string;
  expected_outcome: string;
}

export interface PhasedRoadmapPhase {
  time_frame: string;
  focus_areas: string[];
  key_initiatives: Initiative[];
  dependencies: string[];
}

export interface PhasedRoadmap {
  short_term: PhasedRoadmapPhase;
  mid_term: PhasedRoadmapPhase;
  long_term: PhasedRoadmapPhase;
}

export interface KeyTechnologyEnabler {
  enabler: string;
  impact: string;
}

export interface RiskAndMitigationItem {
  risk: string;
  mitigation: string;
}

export interface InnovationOpportunity {
  idea: string;
  description: string;
  maturity_level: string;
}

export interface TabularSummaryRow {
  time_frame: string;
  key_points: string[];
}

export interface LLMInferredAddition {
  section_title: string;
  content: string;
}

export interface TechnicalRoadmapLLMOutput {
  roadmap_title: string;
  overall_vision: OverallVision;
  current_state_analysis: CurrentStateAnalysis;
  technology_domains: TechnologyDomain[];
  phased_roadmap: PhasedRoadmap;
  key_technology_enablers: KeyTechnologyEnabler[];
  risks_and_mitigations: RiskAndMitigationItem[];
  innovation_opportunities: InnovationOpportunity[];
  tabular_summary: TabularSummaryRow[];
  llm_inferred_additions?: LLMInferredAddition[] | null;
}

export interface TechnicalRoadmapResponse {
  status?: boolean;
  technical_roadmap?: TechnicalRoadmapLLMOutput;
  message?: string;
  error?: string;
  failed?: boolean;
}

// Insights types (mirror backend Pydantic models)
export interface DocumentSummary {
  title: string;
  purpose: string;
  key_themes: string[];
}

export interface KeyDiscussionPoint {
  topic: string;
  details: string;
}

export interface StrengthItem {
  aspect: string;
  evidence_or_example: string;
}

export interface ImprovementOrMissingArea {
  gap: string;
  suggested_improvement: string;
}

export interface FutureConsideration {
  focus_area: string;
  recommendation: string;
}

export interface InnovationAspect {
  innovation_title: string;
  description: string;
  potential_impact: string;
}

export interface PseudocodeOrTechnicalOutline {
  section?: string | null;
  pseudocode?: string | null;
}

export interface InsightsLLMOutput {
  document_summary: DocumentSummary;
  key_discussion_points: KeyDiscussionPoint[];
  strengths: StrengthItem[];
  improvement_or_missing_areas: ImprovementOrMissingArea[];
  future_considerations: FutureConsideration[];
  innovation_aspects: InnovationAspect[];
  pseudocode_or_technical_outline?: PseudocodeOrTechnicalOutline[] | null;
  llm_inferred_additions?: LLMInferredAddition[] | null;
}

export interface InsightsResponse {
  status?: boolean;
  insights?: InsightsLLMOutput;
  message?: string;
  error?: string;
  failed?: boolean;
}

// Strategic Analysis types (mirror backend Pydantic models)
export interface StrategicIntent {
  vision_statement: string;
  stated_objectives: string[];
  implicit_aspirations: string[];
}

export interface StrategicPositioning {
  current_position: string;
  target_position: string;
  competitive_landscape: string;
}

export interface StrategicTheme {
  theme: string;
  description: string;
  evidence_from_document: string;
}

export interface StakeholderInsight {
  stakeholder: string;
  role_or_interest: string;
  influence_level: string;
}

export interface ResourceAndCapability {
  resource: string;
  current_state: string;
  strategic_relevance: string;
}

export interface IdentifiedRisk {
  risk: string;
  severity: string;
  context: string;
}

export interface ForwardLookingAssessment {
  opportunities: string[];
  recommended_next_steps: string[];
  potential_challenges: string[];
  overall_assessment: string;
}

export interface StrategicAnalysisLLMOutput {
  analysis_title: string;
  executive_overview: string;
  strategic_intent: StrategicIntent;
  strategic_positioning: StrategicPositioning;
  key_strategic_themes: StrategicTheme[];
  stakeholder_insights: StakeholderInsight[];
  resources_and_capabilities: ResourceAndCapability[];
  identified_risks: IdentifiedRisk[];
  strategic_gaps_and_observations: string[];
  forward_looking_assessment: ForwardLookingAssessment;
  llm_inferred_additions?: LLMInferredAddition[] | null;
}

export interface StrategicAnalysisResponse {
  status?: boolean;
  strategic_analysis?: StrategicAnalysisLLMOutput;
  message?: string;
  error?: string;
  failed?: boolean;
}

// Technical Analysis types (mirror backend Pydantic models)
export interface TechnicalScope {
  domains_covered: string[];
  technology_stack: string[];
  architecture_overview: string;
}

export interface TechnicalDecision {
  decision: string;
  rationale: string;
  implications: string;
}

export interface TechnicalStrength {
  aspect: string;
  evidence: string;
}

export interface TechnicalConcern {
  concern: string;
  impact: string;
  evidence: string;
}

export interface TechnicalInnovation {
  element: string;
  description: string;
  maturity: string;
}

export interface TechnicalAspirations {
  stated_goals: string[];
  implied_direction: string;
  alignment_assessment: string;
}

export interface ImplementationReadiness {
  ready_components: string[];
  gaps_to_address: string[];
  dependencies: string[];
}

export interface TechnicalForwardAssessment {
  scalability_outlook: string;
  technology_evolution: string;
  recommended_focus_areas: string[];
  overall_assessment: string;
}

export interface TechnicalAnalysisLLMOutput {
  analysis_title: string;
  executive_overview: string;
  technical_scope: TechnicalScope;
  technical_decisions: TechnicalDecision[];
  technical_strengths: TechnicalStrength[];
  technical_concerns: TechnicalConcern[];
  innovation_elements: TechnicalInnovation[];
  technical_aspirations: TechnicalAspirations;
  implementation_readiness: ImplementationReadiness;
  forward_looking_assessment: TechnicalForwardAssessment;
  llm_inferred_additions?: LLMInferredAddition[] | null;
}

export interface TechnicalAnalysisResponse {
  status?: boolean;
  technical_analysis?: TechnicalAnalysisLLMOutput;
  message?: string;
  error?: string;
  failed?: boolean;
}

// ── Tech Sensing types ──

export interface SensingRadarItem {
  name: string;
  quadrant: string;
  ring: string;
  description: string;
  is_new: boolean;
  moved_in?: string | null;
  signal_strength?: number;
  source_count?: number;
  trl?: number;
  patent_count?: number;
}

export interface SensingTrendItem {
  trend_name: string;
  description: string;
  evidence: string[];
  impact_level: string;
  time_horizon: string;
  source_urls?: string[];
}

export interface SensingReportSection {
  section_title: string;
  content: string;
  source_urls?: string[];
}

export interface SensingRecommendation {
  title: string;
  description: string;
  priority: string;
  related_trends: string[];
}

export interface SensingClassifiedArticle {
  title: string;
  source: string;
  url: string;
  published_date: string;
  summary: string;
  relevance_score: number;
  quadrant: string;
  ring: string;
  technology_name: string;
  reasoning: string;
  topic_category?: string;
  industry_segment?: string;
}

export interface SensingRadarItemDetail {
  technology_name: string;
  what_it_is: string;
  why_it_matters: string;
  current_state: string;
  key_players: string[];
  practical_applications: string[];
  source_urls?: string[];
}

export interface SensingHeadlineMove {
  headline: string;
  actor: string;
  segment: string;
  source_urls?: string[];
}

export interface SensingMarketSignal {
  company_or_player: string;
  signal: string;
  strategic_intent: string;
  industry_impact: string;
  segment?: string;
  related_technologies: string[];
  source_urls?: string[];
}

export interface SensingTrendingVideo {
  technology_name: string;
  title: string;
  url: string;
  description: string;
  uploader: string;
  duration: string;
  published: string;
  view_count: number;
  thumbnail_url: string;
}

export interface WeakSignalTrajectoryPoint {
  run_date: string;
  article_count: number;
  source_count: number;
  avg_relevance: number;
  signal_strength: number;
}

export interface WeakSignal {
  technology_name: string;
  current_strength: number;
  acceleration_rate: number;
  first_seen: string;
  run_count: number;
  trajectory: WeakSignalTrajectoryPoint[];
  dvi_score: number;
}

export interface SensingReport {
  report_title: string;
  executive_summary: string;
  domain: string;
  date_range: string;
  total_articles_analyzed: number;
  headline_moves?: SensingHeadlineMove[];
  key_trends: SensingTrendItem[];
  report_sections: SensingReportSection[];
  radar_items: SensingRadarItem[];
  radar_item_details: SensingRadarItemDetail[];
  market_signals: SensingMarketSignal[];
  recommendations: SensingRecommendation[];
  notable_articles: SensingClassifiedArticle[];
  trending_videos?: SensingTrendingVideo[];
  weak_signals?: WeakSignal[];
  relationships?: TechRelationshipMap | null;
}

export interface TechRelationship {
  source_tech: string;
  target_tech: string;
  relationship_type: string;
  strength: number;
  evidence: string;
}

export interface TechCluster {
  cluster_name: string;
  technologies: string[];
  theme: string;
}

export interface TechRelationshipMap {
  relationships: TechRelationship[];
  clusters: TechCluster[];
}

export interface SensingAlert {
  alert_type: string;
  severity: string;
  title: string;
  description: string;
  technology_name: string;
  metadata: Record<string, unknown>;
  timestamp: string;
}

export interface AlertPreferences {
  enabled: boolean;
  email_alerts: boolean;
  ring_jump_threshold: number;
  weak_signal_acceleration_threshold: number;
  alert_on_direct_adopt: boolean;
  alert_on_stack_match: boolean;
  alert_on_trend_surge: boolean;
}

export interface SensingReportData {
  report: SensingReport;
  meta: {
    tracking_id: string;
    domain: string;
    raw_article_count: number;
    deduped_article_count: number;
    classified_article_count: number;
    execution_time_seconds: number;
    generated_at: string;
    custom_requirements?: string;
    must_include?: string[] | null;
    dont_include?: string[] | null;
    lookback_days?: number;
    alerts?: SensingAlert[];
  };
}

export interface SensingHistoryItem {
  tracking_id: string;
  domain: string;
  generated_at: string;
  report_title: string;
  total_articles: number;
  custom_requirements?: string;
  must_include?: string[] | null;
  dont_include?: string[] | null;
  lookback_days?: number;
}

// ── Sensing Schedule types ──

export interface SensingSchedule {
  id: string;
  user_id: string;
  domain: string;
  frequency: string;
  custom_requirements: string;
  must_include?: string[] | null;
  dont_include?: string[] | null;
  lookback_days: number;
  enabled: boolean;
  created_at: string;
  next_run: string;
  last_run?: string | null;
}

// ── Sensing Comparison types ──

export interface RadarDiffItem {
  name: string;
  status: 'added' | 'removed' | 'moved' | 'unchanged';
  quadrant: string;
  current_ring?: string | null;
  previous_ring?: string | null;
  description: string;
}

export interface TrendDiff {
  name: string;
  status: 'new' | 'removed' | 'continuing';
}

export interface ReportComparison {
  report_a_id: string;
  report_b_id: string;
  report_a_title: string;
  report_b_title: string;
  report_a_date: string;
  report_b_date: string;
  radar_diff: RadarDiffItem[];
  trend_diff: TrendDiff[];
  new_signals: string[];
  removed_signals: string[];
  summary: string;
}

// ── Sensing Timeline types ──

export interface TechnologyTimelineEntry {
  report_date: string;
  report_id: string;
  ring: string;
  quadrant: string;
}

export interface TechnologyTimeline {
  technology_name: string;
  quadrant: string;
  entries: TechnologyTimelineEntry[];
}

export interface TimelineData {
  domain: string;
  technologies: TechnologyTimeline[];
}

// ── Sensing Org Context types ──

export interface RadarQuadrantConfig {
  name: string;
  color: string;
}

export interface RadarCustomization {
  quadrants: RadarQuadrantConfig[];
}

export interface OrgTechContext {
  tech_stack: string[];
  industry: string;
  priorities: string[];
  radar_customization?: RadarCustomization | null;
}

// ── Sensing Deep Dive types ──

export interface CompetitorEntry {
  name: string;
  approach: string;
  strengths: string;
  weaknesses: string;
}

export interface KeyResource {
  title: string;
  url: string;
  type: string;
}

export interface DeepDiveReport {
  technology_name: string;
  comprehensive_analysis: string;
  technical_architecture: string;
  competitive_landscape: CompetitorEntry[];
  adoption_roadmap: string;
  risk_assessment: string;
  key_resources: KeyResource[];
  recommendations: string[];
}

// ── Sensing Deep Dive Follow-Up types ──

export interface DeepDiveFollowUpResponse {
  answer: string;
  sources_used: string[];
  suggested_questions: string[];
}

// ── Sensing Deep Dive History types ──

export interface DeepDiveHistoryItem {
  tracking_id: string;
  technology_name: string;
  domain: string;
  generated_at: string;
  message_count: number;
}

export interface DeepDiveFullLoad {
  report: DeepDiveReport;
  conversation_history: { role: string; content: string }[];
  meta: { tracking_id: string; technology_name: string; domain: string; generated_at: string };
}

// ── Sensing Collaboration types ──

export interface RadarVote {
  vote_id: string;
  user_id: string;
  user_name: string;
  radar_item_name: string;
  suggested_ring: string;
  reasoning: string;
  created_at: string;
}

export interface RadarComment {
  comment_id: string;
  user_id: string;
  user_name: string;
  radar_item_name: string;
  text: string;
  created_at: string;
}

export interface SharedReport {
  share_id: string;
  report_tracking_id: string;
  owner_user_id: string;
  votes: RadarVote[];
  comments: RadarComment[];
  created_at: string;
}

export interface SharedReportFeedback {
  share_id: string;
  votes: RadarVote[];
  comments: RadarComment[];
  vote_summary: Record<string, { votes: RadarVote[]; ring_counts: Record<string, number> }>;
  total_votes: number;
  total_comments: number;
}

// ── Excel Skill types ──

export interface ExcelSkillGenerateResponse {
  status: boolean;
  message?: string;
  tracking_id?: string;
}

export interface ExcelSkillStatusResponse {
  status: boolean;
  message?: string;
  failed?: boolean;
  error?: string;
  result?: {
    file_name: string;
    download_url: string;
    description: string;
    sheet_count: number;
    total_rows: number;
  };
}

export interface ExcelSkillListItem {
  tracking_id: string;
  file_name: string;
  download_url: string;
  description: string;
  sheet_count: number;
  total_rows: number;
  created_at: string;
  request_text: string;
}

// ── Document Creator types ──

export type DocumentType =
  | 'presentation'
  | 'executive_summary'
  | 'technical_report'
  | 'research_brief'
  | 'project_proposal'
  | 'comparison_report';

export type AudienceType = 'executive' | 'technical' | 'general';
export type ToneType = 'formal' | 'professional' | 'conversational' | 'academic';
export type ContentFormat = 'prose' | 'bullets' | 'table' | 'mixed';
export type SectionStatus =
  | 'pending'
  | 'generating'
  | 'generated'
  | 'approved'
  | 'needs_revision'
  | 'failed';
export type ExportFormat = 'pptx' | 'docx' | 'pdf';

export interface DocumentCreatorConfig {
  document_type: DocumentType;
  audience: AudienceType;
  tone: ToneType;
  source_document_ids?: string[] | null;
  custom_instructions?: string | null;
  length_preference?: string;
}

export interface OutlineSectionSpec {
  section_id: string;
  title: string;
  description: string;
  content_format: ContentFormat;
  heading_level: number;
  guidance?: string | null;
  source_document_ids: string[];
  order: number;
}

export interface SectionVersion {
  version_id: string;
  content: string;
  bullet_points?: string[] | null;
  table_data?: { headers: string[]; rows: string[][] } | null;
  speaker_notes?: string | null;
  key_takeaway?: string | null;
  sources_used: Array<{ document_id: string; title: string; page_no: number }>;
  feedback_used?: string | null;
  generated_at: string;
}

export interface SectionState {
  spec: OutlineSectionSpec;
  status: SectionStatus;
  versions: SectionVersion[];
  selected_version_index: number;
}

export interface DocumentCreatorOutlineResponse {
  status: boolean;
  tracking_id?: string;
  message?: string;
  error?: string;
}

export interface OutlineStatusResponse {
  status: boolean;
  message?: string;
  failed?: boolean;
  error?: string;
  outline?: {
    doc_gen_id: string;
    document_title: string;
    document_subtitle?: string;
    sections: Array<{
      section_id: string;
      title: string;
      description: string;
      content_format: string;
      order: number;
    }>;
  };
}

export interface GenerationStatusResponse {
  doc_gen_id: string;
  phase: string;
  sections: Array<{
    section_id: string;
    title: string;
    status: string;
  }>;
  completed_count: number;
  total_count: number;
}

export interface DocumentCreatorPreviewResponse {
  status: string;
  doc_gen_id: string;
  document_title: string;
  document_subtitle?: string;
  config: DocumentCreatorConfig;
  sections: SectionState[];
  review_result?: {
    overall_score: number;
    coherence_score: number;
    completeness_score: number;
    consistency_score: number;
    issues: string[];
    approved: boolean;
  } | null;
}

export interface DocumentCreatorExportResponse {
  status: string;
  filename?: string;
  download_url?: string;
  error?: string;
}

export interface DocumentCreatorListItem {
  doc_gen_id: string;
  document_title: string;
  phase: string;
  document_type: string;
  section_count: number;
  created_at: string;
  updated_at: string;
}

// Auth helpers
export const getAuthToken = () => localStorage.getItem('auth_token');
export const setAuthToken = (token: string) => localStorage.setItem('auth_token', token);
export const removeAuthToken = () => localStorage.removeItem('auth_token');
export const getCurrentUser = (): User | null => {
  const userStr = localStorage.getItem('current_user');
  return userStr ? JSON.parse(userStr) : null;
};
export const setCurrentUser = (user: User) => localStorage.setItem('current_user', JSON.stringify(user));
export const removeCurrentUser = () => localStorage.removeItem('current_user');

// API functions
export const api = {
  async register(name: string, email: string, password: string) {
    const response = await fetch(`${API_URL}/user/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password }),
    });
    return response;
  },

  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await fetch(`${API_URL}/user/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = errorData.detail || 'Invalid email or password';
      throw new Error(errorMessage);
    }

    return response.json();
  },

  async getUser(userId: string): Promise<User> {
    const token = getAuthToken();
    console.log("Using token:", token);
    console.log("Fetching user with ID:", userId);
    const response = await fetch(`${API_URL}/user/${userId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    return data.user;
  },

  async uploadFiles(data: { thread_name?: string; thread_id?: string; files: File[] }): Promise<UploadResponse> {
    const token = getAuthToken();
    const formData = new FormData();

    if (data.thread_name) formData.append('thread_name', data.thread_name);
    if (data.thread_id) formData.append('thread_id', data.thread_id);
    data.files.forEach(file => formData.append('files', file));

    const response = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });

    const json = await response.json();

    if (!response.ok || json.error) {
      const errorMessage = json.error || json.detail || 'Failed to upload files';
      throw new Error(errorMessage);
    }

    return json;
  },


  async uploadFilesWithProgress(params: {
    thread_name?: string;
    thread_id?: string;
    files: File[];
    onProgress?: (args: { fileIndex: number; loaded: number; total: number; percent: number }) => void;
  }): Promise<UploadResponse> {
    const token = getAuthToken();

    const uploadSingle = (file: File): Promise<UploadResponse> => {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_URL}/upload`, true);
        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }

        xhr.onload = () => {
          try {
            const json = JSON.parse(xhr.responseText);
            resolve(json);
          } catch (e) {
            reject(e);
          }
        };

        xhr.onerror = () => reject(new Error('Network error during upload'));

        const formData = new FormData();
        if (params.thread_name) formData.append('thread_name', params.thread_name);
        if (params.thread_id) formData.append('thread_id', params.thread_id);
        formData.append('files', file);

        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable && params.onProgress) {
            const percent = Math.round((event.loaded / event.total) * 100);
          }
        };

        xhr.send(formData);
      });
    };

    const results: UploadResponse = {
      status: 'success',
      message: 'Uploaded',
      thread_id: params.thread_id || '',
      documents: [],
    };

    for (let i = 0; i < params.files.length; i++) {
      const file = params.files[i];
      await new Promise<UploadResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_URL}/upload`, true);
        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }

        xhr.onload = () => {
          try {
            const json: UploadResponse = JSON.parse(xhr.responseText);

            // Check for errors in response
            if (xhr.status >= 400 || (json as any).error) {
              const errorMessage = (json as any).error || (json as any).detail || 'Failed to upload file';
              reject(new Error(errorMessage));
              return;
            }

            if (!results.thread_id && json.thread_id) {
              results.thread_id = json.thread_id;
            }
            results.documents = [...results.documents, ...json.documents];
            resolve(json);
          } catch (e) {
            reject(e);
          }
        };

        xhr.onerror = () => reject(new Error('Network error during upload'));

        const formData = new FormData();
        if (params.thread_name) formData.append('thread_name', params.thread_name);
        if (params.thread_id) formData.append('thread_id', params.thread_id);
        formData.append('files', file);

        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable && params.onProgress) {
            const percent = Math.round((event.loaded / event.total) * 100);
            params.onProgress({ fileIndex: i, loaded: event.loaded, total: event.total, percent });
          }
        };

        xhr.send(formData);
      });
    }

    return results;
  },

  async getThread(threadId: string): Promise<Thread> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      const errorMessage = data.error || data.detail || 'Failed to load thread';
      throw new Error(errorMessage);
    }

    return data.thread;
  },

  async query(
    threadId: string,
    question: string,
    mode: 'Internal' | 'External',
    useSelfKnowledge: boolean,
    useContext: boolean = false
  ): Promise<QueryResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        thread_id: threadId,
        question,
        mode,
        use_self_knowledge: useSelfKnowledge,
        use_context: useContext,
      }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      const errorMessage = data.error || data.detail || 'Failed to get response';
      throw new Error(errorMessage);
    }

    return data;
  },

  async deleteThread(threadId: string): Promise<{ status: boolean }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    return response.json();
  },

  async deleteDocument(threadId: string, docId: string): Promise<DeleteDocumentResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}/document/${docId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = errorData.error || errorData.detail || 'Failed to delete document';
      throw new Error(errorMessage);
    }

    return response.json();
  },

  async addExistingDocument(
    targetThreadId: string,
    sourceThreadId: string,
    docId: string,
  ): Promise<AddExistingDocumentResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/thread/${targetThreadId}/documents/add-existing`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          source_thread_id: sourceThreadId,
          doc_id: docId,
        }),
      },
    );

    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || data.detail || 'Failed to add document');
    }
    return data;
  },

  async updateThread(threadId: string, data: { thread_name: string }): Promise<{ status: string; message: string; thread_id: string; thread_name: string }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(data),
    });

    return response.json();
  },

  async deleteChat(threadId: string, chatIndex: number): Promise<DeleteChatResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}/chats/${chatIndex}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = errorData.error || errorData.detail || 'Failed to delete chat';
      throw new Error(errorMessage);
    }

    return response.json();
  },

  async clearThreadChats(threadId: string): Promise<ClearChatsResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}/chats`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = errorData.error || errorData.detail || 'Failed to clear chats';
      throw new Error(errorMessage);
    }

    return response.json();
  },

  async getMindMap(threadId: string): Promise<MindMapResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/mindmap/${threadId}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    if (!response.ok) {
      // Fallback shape with mind_map=false so UI can display error message
      return { mind_map: false, message: `Failed to fetch mind map (${response.status})` };
    }
    return response.json();
  },

  async summary(threadId: string, documentId: string, regenerate: boolean = false): Promise<SummaryResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/summary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse summary response (${response.status})` };
    }
    if (!response.ok) {
      // Normalize error shape
      return { status: false, error: data?.detail || data?.message || 'Summary request failed' };
    }
    return data as SummaryResponse;
  },

  async summaryGlobal(threadId: string, regenerate: boolean = false): Promise<SummaryResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/summary/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse summary (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Summary (global) request failed' };
    }
    return data as SummaryResponse;
  },

  async strategicRoadmap(threadId: string, documentId: string, regenerate: boolean = false): Promise<StrategicRoadmapResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/strategic_roadmap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse strategic roadmap response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Strategic roadmap request failed' };
    }
    // Expected shapes: { status: false, message } or { status: true, strategic_roadmap }
    return data as StrategicRoadmapResponse;
  },

  // Generate strategic roadmap across ALL documents in a thread
  async strategicRoadmapGlobal(threadId: string, regenerate: boolean = false): Promise<StrategicRoadmapResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/strategic_roadmap/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse strategic roadmap (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Strategic roadmap (global) request failed' };
    }
    return data as StrategicRoadmapResponse;
  },

  async insights(threadId: string, documentId: string, regenerate: boolean = false): Promise<InsightsResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/insights`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse insights response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Insights request failed' };
    }
    return data as InsightsResponse; // Expected: { status:false,message } or { status:true, insights }
  },

  async insightsGlobal(threadId: string, regenerate: boolean = false): Promise<InsightsResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/insights/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse insights (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Insights (global) request failed' };
    }
    return data as InsightsResponse;
  },

  async technicalRoadmap(threadId: string, documentId: string, regenerate: boolean = false): Promise<TechnicalRoadmapResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/technical_roadmap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse technical roadmap response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Technical roadmap request failed' };
    }
    return data as TechnicalRoadmapResponse;
  },

  async technicalRoadmapGlobal(threadId: string, regenerate: boolean = false): Promise<TechnicalRoadmapResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/technical_roadmap/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse technical roadmap (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Technical roadmap (global) request failed' };
    }
    return data as TechnicalRoadmapResponse;
  },

  // ── Strategic Analysis ──

  async strategicAnalysis(threadId: string, documentId: string, regenerate: boolean = false): Promise<StrategicAnalysisResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/strategic_analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse strategic analysis response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Strategic analysis request failed' };
    }
    return data as StrategicAnalysisResponse;
  },

  async strategicAnalysisGlobal(threadId: string, regenerate: boolean = false): Promise<StrategicAnalysisResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/strategic_analysis/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse strategic analysis (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Strategic analysis (global) request failed' };
    }
    return data as StrategicAnalysisResponse;
  },

  // ── Technical Analysis ──

  async technicalAnalysis(threadId: string, documentId: string, regenerate: boolean = false): Promise<TechnicalAnalysisResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/technical_analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, document_id: documentId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse technical analysis response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Technical analysis request failed' };
    }
    return data as TechnicalAnalysisResponse;
  },

  async technicalAnalysisGlobal(threadId: string, regenerate: boolean = false): Promise<TechnicalAnalysisResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/technical_analysis/global`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_id: threadId, regenerate }),
    });
    let data: any = null;
    try {
      data = await response.json();
    } catch (_) {
      return { status: false, error: `Failed to parse technical analysis (global) response (${response.status})` };
    }
    if (!response.ok) {
      return { status: false, error: data?.detail || data?.message || 'Technical analysis (global) request failed' };
    }
    return data as TechnicalAnalysisResponse;
  },

  // ── Thread Instructions ──

  async getInstructions(threadId: string): Promise<ThreadInstruction[]> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}/instructions`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Failed to load instructions');
    }
    return data.instructions ?? [];
  },

  async addInstruction(threadId: string, text: string): Promise<ThreadInstruction> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/thread/${threadId}/instructions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Failed to add instruction');
    }
    return data.instruction;
  },

  async updateInstruction(
    threadId: string,
    instructionId: string,
    updates: { text?: string; selected?: boolean }
  ): Promise<ThreadInstruction> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/thread/${threadId}/instructions/${instructionId}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(updates),
      }
    );
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Failed to update instruction');
    }
    return data.instruction;
  },

  async deleteInstruction(threadId: string, instructionId: string): Promise<void> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/thread/${threadId}/instructions/${instructionId}`,
      {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Failed to delete instruction');
    }
  },

  // ── Document Creator ──

  async documentCreatorOutline(
    threadId: string,
    config: DocumentCreatorConfig,
  ): Promise<DocumentCreatorOutlineResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/document-creator/outline`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        thread_id: threadId,
        document_type: config.document_type,
        audience: config.audience,
        tone: config.tone,
        source_document_ids: config.source_document_ids,
        custom_instructions: config.custom_instructions,
        length_preference: config.length_preference || 'medium',
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to generate outline');
    }
    return data;
  },

  async documentCreatorOutlineStatus(
    trackingId: string,
  ): Promise<OutlineStatusResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/outline-status/${trackingId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to check outline status');
    }
    return data;
  },

  async documentCreatorUpdateOutline(
    docGenId: string,
    sections: OutlineSectionSpec[],
    documentTitle?: string,
    documentSubtitle?: string,
  ): Promise<{ status: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/outline/${docGenId}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          sections,
          document_title: documentTitle,
          document_subtitle: documentSubtitle,
        }),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to update outline');
    }
    return data;
  },

  async documentCreatorGenerate(
    docGenId: string,
  ): Promise<{ status: string; message: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/generate/${docGenId}`,
      {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to start generation');
    }
    return data;
  },

  async documentCreatorStatus(
    docGenId: string,
  ): Promise<GenerationStatusResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/status/${docGenId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to check generation status');
    }
    return data;
  },

  async documentCreatorPreview(
    docGenId: string,
  ): Promise<DocumentCreatorPreviewResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/preview/${docGenId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to load preview');
    }
    return data;
  },

  async documentCreatorIterate(
    docGenId: string,
    sectionId: string,
    feedback?: string,
  ): Promise<{ status: string; message: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/iterate/${docGenId}/${sectionId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ feedback: feedback || null }),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to iterate section');
    }
    return data;
  },

  async documentCreatorSelectVersion(
    docGenId: string,
    sectionId: string,
    versionIndex: number,
  ): Promise<{ status: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/select-version/${docGenId}/${sectionId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ version_index: versionIndex }),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to select version');
    }
    return data;
  },

  async documentCreatorApprove(
    docGenId: string,
    sectionId: string,
  ): Promise<{ status: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/approve/${docGenId}/${sectionId}`,
      {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to approve section');
    }
    return data;
  },

  async documentCreatorEditSection(
    docGenId: string,
    sectionId: string,
    edits: {
      content?: string;
      bullet_points?: string[];
      key_takeaway?: string;
      speaker_notes?: string;
    },
  ): Promise<{ section_id: string; status: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/edit-section/${docGenId}/${sectionId}`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(edits),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to edit section');
    }
    return data;
  },

  async documentCreatorExport(
    docGenId: string,
    format: ExportFormat,
  ): Promise<DocumentCreatorExportResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/export/${docGenId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ format }),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to export document');
    }
    return data;
  },

  async documentCreatorDownload(
    docGenId: string,
    filename: string,
  ): Promise<Blob> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/download/${docGenId}/${filename}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    if (!response.ok) {
      throw new Error('Failed to download file');
    }
    return response.blob();
  },

  async documentCreatorList(
    threadId: string,
  ): Promise<{ documents: DocumentCreatorListItem[] }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/list/${threadId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to list documents');
    }
    return data;
  },

  async documentCreatorDelete(
    docGenId: string,
  ): Promise<{ status: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/document-creator/${docGenId}`,
      {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to delete document');
    }
    return data;
  },

  // ── Settings ──

  async getSwitches(): Promise<{ switches: Record<string, boolean> }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/settings/switches`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to fetch switches');
    }
    return data;
  },

  async updateSwitch(key: string, value: boolean): Promise<{ key: string; value: boolean }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/settings/switches/${key}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ value }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to update switch');
    }
    return data;
  },

  // ── Excel Skill ──

  async excelSkillGenerate(
    threadId: string,
    requestText: string,
    sourceDocumentIds?: string[],
  ): Promise<ExcelSkillGenerateResponse> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/excel-skill/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        thread_id: threadId,
        request_text: requestText,
        source_document_ids: sourceDocumentIds || null,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to start Excel generation');
    }
    return data;
  },

  async excelSkillStatus(
    trackingId: string,
  ): Promise<ExcelSkillStatusResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/excel-skill/status/${trackingId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Failed to check Excel status');
    }
    return data;
  },

  async excelSkillDownload(
    threadId: string,
    filename: string,
  ): Promise<Blob> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/excel-skill/download/${threadId}/${filename}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    if (!response.ok) {
      throw new Error('Failed to download Excel file');
    }
    return response.blob();
  },

  async excelSkillList(
    threadId: string,
  ): Promise<{ files: ExcelSkillListItem[] }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/excel-skill/list/${threadId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to list Excel files');
    }
    return data;
  },

  async excelSkillDelete(
    threadId: string,
    trackingId: string,
  ): Promise<void> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/excel-skill/${threadId}/${trackingId}`,
      { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } },
    );
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || 'Failed to delete Excel file');
    }
  },

  // --- Tech Sensing ---

  async sensingGenerate(
    domain: string = 'Generative AI',
    customRequirements: string = '',
    mustInclude?: string[],
    dontInclude?: string[],
    lookbackDays: number = 7,
    feedUrls?: string[],
    searchQueries?: string[],
  ): Promise<{ status: string; tracking_id: string; message: string }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        domain,
        custom_requirements: customRequirements,
        must_include: mustInclude?.length ? mustInclude : null,
        dont_include: dontInclude?.length ? dontInclude : null,
        lookback_days: lookbackDays,
        feed_urls: feedUrls || null,
        search_queries: searchQueries || null,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to start sensing report generation');
    }
    return data;
  },

  async sensingGenerateFromDocument(
    file: File,
    domain: string = 'Generative AI',
    customRequirements: string = '',
    mustInclude?: string[],
    dontInclude?: string[],
  ): Promise<{ status: string; tracking_id: string; message: string }> {
    const token = getAuthToken();
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domain', domain);
    formData.append('custom_requirements', customRequirements);
    if (mustInclude?.length) formData.append('must_include', mustInclude.join(','));
    if (dontInclude?.length) formData.append('dont_include', dontInclude.join(','));

    const response = await fetch(`${API_URL}/sensing/generate-from-document`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to start document sensing');
    }
    return data;
  },

  async sensingStatus(
    trackingId: string,
  ): Promise<{ status: string; data?: SensingReportData; error?: string }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/status/${trackingId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to check sensing status');
    }
    return data;
  },

  async sensingHistory(): Promise<{ reports: SensingHistoryItem[] }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/history`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to load sensing history');
    }
    return data;
  },

  async sensingDelete(reportId: string): Promise<void> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/report/${reportId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || 'Failed to delete sensing report');
    }
  },

  async sensingCompare(tidA: string, tidB: string): Promise<ReportComparison> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/compare?a=${encodeURIComponent(tidA)}&b=${encodeURIComponent(tidB)}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to compare reports');
    }
    return data;
  },

  async sensingCreateSchedule(params: {
    domain: string; frequency: string; custom_requirements?: string;
    must_include?: string[] | null; dont_include?: string[] | null; lookback_days?: number;
  }): Promise<SensingSchedule> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/schedule`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to create schedule');
    return data;
  },

  async sensingGetSchedules(): Promise<{ schedules: SensingSchedule[] }> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/schedules`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load schedules');
    return data;
  },

  async sensingUpdateSchedule(id: string, updates: Record<string, unknown>): Promise<SensingSchedule> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/schedule/${id}`, {
      method: 'PUT',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to update schedule');
    return data;
  },

  async sensingDeleteSchedule(id: string): Promise<void> {
    const token = getAuthToken();
    const response = await fetch(`${API_URL}/sensing/schedule/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || 'Failed to delete schedule');
    }
  },

  async sensingGetFeeds(domain: string): Promise<{ feeds: string[]; queries: string[] }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/feeds?domain=${encodeURIComponent(domain)}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load feeds');
    return data;
  },

  async sensingTimeline(domain?: string): Promise<TimelineData> {
    const token = getAuthToken();
    const params = domain ? `?domain=${encodeURIComponent(domain)}` : '';
    const response = await fetch(
      `${API_URL}/sensing/timeline${params}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load timeline');
    return data;
  },

  async sensingGetOrgContext(): Promise<OrgTechContext> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/org-context`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load org context');
    return data;
  },

  async sensingUpdateOrgContext(context: OrgTechContext): Promise<OrgTechContext> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/org-context`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(context),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to update org context');
    return data;
  },

  async sensingDeepDive(technologyName: string, domain: string): Promise<{ status: string; tracking_id: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/deep-dive`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ technology_name: technologyName, domain }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to start deep dive');
    return data;
  },

  async sensingDeepDiveStatus(trackingId: string): Promise<{ status: string; data?: DeepDiveReport; error?: string }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/deep-dive/status/${trackingId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to get deep dive status');
    return data;
  },

  async sensingDeepDiveFollowUp(
    technologyName: string,
    domain: string,
    question: string,
    trackingId: string,
  ): Promise<DeepDiveFollowUpResponse> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/deep-dive-followup`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ technology_name: technologyName, domain, question, tracking_id: trackingId }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to get follow-up');
    return data;
  },

  async sensingDeepDiveHistory(): Promise<{ deep_dives: DeepDiveHistoryItem[] }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/deep-dive/history`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load deep dive history');
    return data;
  },

  async sensingDeepDiveLoad(trackingId: string): Promise<DeepDiveFullLoad> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/deep-dive/${trackingId}/full`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load deep dive');
    return data;
  },

  async sensingShare(reportId: string): Promise<SharedReport> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/share/${reportId}`,
      { method: 'POST', headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to share report');
    return data;
  },

  async sensingGetShared(shareId: string): Promise<{ shared: SharedReport; report: SensingReportData | null }> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/shared/${shareId}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load shared report');
    return data;
  },

  async sensingVote(shareId: string, radarItemName: string, suggestedRing: string, reasoning?: string): Promise<RadarVote> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/shared/${shareId}/vote`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ radar_item_name: radarItemName, suggested_ring: suggestedRing, reasoning: reasoning || '' }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to submit vote');
    return data;
  },

  async sensingComment(shareId: string, text: string, radarItemName?: string): Promise<RadarComment> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/shared/${shareId}/comment`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ text, radar_item_name: radarItemName || '' }),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to add comment');
    return data;
  },

  async sensingGetAlertPrefs(): Promise<AlertPreferences> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/alert-prefs`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load alert preferences');
    return data;
  },

  async sensingUpdateAlertPrefs(prefs: AlertPreferences): Promise<AlertPreferences> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/alert-prefs`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(prefs),
      },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to update alert preferences');
    return data;
  },

  async sensingGetFeedback(shareId: string): Promise<SharedReportFeedback> {
    const token = getAuthToken();
    const response = await fetch(
      `${API_URL}/sensing/shared/${shareId}/feedback`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Failed to load feedback');
    return data;
  },

};

// WebSocket helper
export const getWebSocketUrl = (path: string) => {
  // Allow explicit WS base via env, otherwise derive from API_URL
  const base = (import.meta.env.VITE_WS_URL as string | undefined) || API_URL;
  const url = new URL(base);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  // Ensure we don't double up slashes
  const joined = `${url.origin}${path.startsWith('/') ? '' : '/'}${path}`;
  return joined;
};
