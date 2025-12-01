import React, { useState } from 'react';
import axios from 'axios';
import { Search, BarChart2, History, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { Link } from 'react-router-dom';

// Types for our data structures
interface SearchResult {
  doc_id: string;
  score: number;
  text: string;
}

interface Metrics {
  baseline: { ndcg: number; snips: number };
  target: { ndcg: number; snips: number };
  lift: { ndcg: number; snips: number };
}

interface SimulationLog {
  rank: number;
  click: number;
  propensity: number;
  relevance_prob: number;
  doc_text: string;
}

interface SearchResponse {
  results: SearchResult[];
  metrics: Metrics;
  simulation_logs: SimulationLog[];
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:8000/api/search', { query });
      setResults(response.data);
    } catch (err) {
      setError('Failed to fetch results. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // If we have results, the search bar is at top. If not, it's centered.
  const isCentered = !results && !loading;

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 flex flex-col">
      
      {/* Search Bar Section - Dynamic Positioning */}
      <div className={`transition-all duration-500 ease-in-out ${isCentered ? 'flex-grow flex flex-col justify-center items-center -mt-20' : 'bg-white shadow-sm py-4 sticky top-0 z-10'}`}>
        <div className={`container mx-auto px-4 ${isCentered ? 'max-w-2xl text-center' : 'max-w-6xl flex items-center gap-6'}`}>
          
          {/* Logo */}
          <h1 className={`font-bold text-blue-600 transition-all duration-300 ${isCentered ? 'text-6xl mb-8' : 'text-2xl flex-shrink-0'}`}>
            ULTR Search
          </h1>

          {/* Form */}
          <form onSubmit={handleSearch} className={`relative transition-all duration-300 ${isCentered ? 'w-full' : 'flex-grow max-w-3xl'}`}>
            <div className="relative group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for anything..."
                className={`w-full px-6 rounded-full border border-gray-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${isCentered ? 'py-4 text-lg shadow-md hover:shadow-lg' : 'py-2.5 text-base bg-gray-50 hover:bg-white hover:shadow-md focus:bg-white'}`}
              />
              <button 
                type="submit" 
                className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 text-blue-600 rounded-full hover:bg-blue-50 transition-colors ${loading ? 'opacity-50' : ''}`}
                disabled={loading}
              >
                {loading ? (
                  <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
                ) : (
                  <Search size={isCentered ? 24 : 20} />
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Main Content Area (Only shown when results exist) */}
      {results && (
        <div className="container mx-auto px-4 max-w-6xl py-8 flex flex-col lg:flex-row gap-8 animate-fade-in">
          
          {/* Left Column: Search Results */}
          <div className="lg:w-2/3 space-y-6">
            <div className="flex items-baseline justify-between border-b border-gray-200 pb-2 mb-4">
              <h2 className="text-sm font-medium text-gray-500">
                Top Results for "<span className="text-gray-800">{query}</span>"
              </h2>
              <span className="text-xs text-gray-400">
                Ranked by Cross-Encoder
              </span>
            </div>

            {results.results.map((result, index) => (
              <div key={index} className="group bg-white rounded-xl p-5 border border-transparent hover:border-blue-100 hover:shadow-md transition-all duration-200">
                <div className="flex justify-between items-start gap-4 mb-1">
                  <Link 
                    to={`/doc/${result.doc_id}`} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-xl text-blue-600 hover:underline font-medium leading-snug"
                  >
                    Document {result.doc_id}
                  </Link>
                  <span className="shrink-0 text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                    Score: {result.score.toFixed(4)}
                  </span>
                </div>
                <div className="text-green-700 text-xs mb-2 flex items-center gap-1">
                  <Link to={`/doc/${result.doc_id}`} target="_blank" rel="noopener noreferrer" className="hover:underline">
                    www.msmarco-passage.com/doc/{result.doc_id}
                  </Link>
                  <span className="w-0.5 h-0.5 bg-green-700 rounded-full mx-1"></span>
                  <span className="text-gray-400">Cached</span>
                </div>
                <p className="text-gray-600 text-sm leading-relaxed line-clamp-3">
                  {result.text}
                </p>
              </div>
            ))}
          </div>

          {/* Right Column: Simulation & Evaluation (Sidebar) */}
          <div className="lg:w-1/3 space-y-6">
            
            {/* Metrics Card */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden sticky top-24">
              <div className="bg-gray-50 px-5 py-4 border-b border-gray-200 flex items-center gap-2">
                <BarChart2 size={18} className="text-blue-600" />
                <h3 className="font-semibold text-gray-800">OPE Evaluation</h3>
              </div>
              
              <div className="p-5 space-y-6">
                <MetricRow 
                  title="Oracle nDCG" 
                  baseline={results.metrics.baseline.ndcg}
                  target={results.metrics.target.ndcg}
                  lift={results.metrics.lift.ndcg}
                  description="True ranking quality vs baseline"
                />
                
                <div className="border-t border-gray-100 pt-4"></div>
                
                <MetricRow 
                  title="SNIPS (Counterfactual)" 
                  baseline={results.metrics.baseline.snips}
                  target={results.metrics.target.snips}
                  lift={results.metrics.lift.snips}
                  description="Estimated value from biased logs"
                />
              </div>

              {/* Simulation Logs Accordion */}
              <div className="border-t border-gray-200">
                <button 
                  onClick={() => setShowLogs(!showLogs)}
                  className="w-full flex items-center justify-between px-5 py-3 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <History size={16} />
                    <span>Simulation Logs</span>
                  </span>
                  {showLogs ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>

                {showLogs && (
                  <div className="bg-gray-50 max-h-96 overflow-y-auto border-t border-gray-200">
                    {results.simulation_logs.map((log, idx) => (
                      <div key={idx} className="px-5 py-3 border-b border-gray-100 text-xs hover:bg-white transition-colors">
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-bold text-gray-700">Rank {log.rank}</span>
                          {log.click ? (
                            <span className="text-green-600 flex items-center gap-1 font-medium bg-green-50 px-1.5 py-0.5 rounded"><CheckCircle size={10}/> Click</span>
                          ) : (
                            <span className="text-gray-400 flex items-center gap-1 bg-gray-100 px-1.5 py-0.5 rounded"><XCircle size={10}/> Skip</span>
                          )}
                        </div>
                        <div className="flex justify-between text-gray-500 mb-1">
                          <span>P(Examine): {log.propensity.toFixed(2)}</span>
                          <span>P(Rel): {log.relevance_prob.toFixed(2)}</span>
                        </div>
                        <div className="text-gray-600 truncate font-mono bg-white p-1 rounded border border-gray-200">
                          {log.doc_text}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>
      )}

      {/* Footer for empty state */}
      {isCentered && (
        <div className="text-center pb-8 text-sm text-gray-400">
          Powered by Unbiased Learning to Rank (ULTR) • ms-marco-MiniLM-L-6-v2
        </div>
      )}
    </div>
  );
};

interface MetricRowProps {
  title: string;
  baseline: number;
  target: number;
  lift: number;
  description: string;
}

const MetricRow: React.FC<MetricRowProps> = ({ title, baseline, target, lift, description }) => {
  const isPositive = lift > 0;
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="font-medium text-gray-700">{title}</span>
        <span className={`text-xs font-bold px-2 py-1 rounded-full ${isPositive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
          {isPositive ? '+' : ''}{lift.toFixed(1)}% Lift
        </span>
      </div>
      
      <div className="flex items-center gap-2 mb-1">
        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-gray-400" style={{ width: `${Math.min(baseline * 100, 100)}%` }}></div>
        </div>
        <span className="text-xs text-gray-500 w-12 text-right">{baseline.toFixed(3)}</span>
      </div>
      
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500" style={{ width: `${Math.min(target * 100, 100)}%` }}></div>
        </div>
        <span className="text-xs text-blue-600 font-bold w-12 text-right">{target.toFixed(3)}</span>
      </div>
      
      <p className="text-xs text-gray-400 mt-2">{description}</p>
    </div>
  );
};

export default SearchPage;
