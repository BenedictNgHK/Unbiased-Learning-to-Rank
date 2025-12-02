import React, { useState } from 'react';
import axios from 'axios';
import { Search, BarChart2, History, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { Link } from 'react-router-dom';

interface SearchResult { doc_id: string; score: number; text: string; }
interface Pair { ndcg: number; snips: number; }
interface MetricsLegacy { baseline: Pair; target: Pair; lift: { ndcg: number; snips: number }; }
interface SimulationLog { rank: number; click: number; propensity: number; relevance_prob: number; doc_text: string; }
interface SearchResponse {
  results: SearchResult[];
  metrics: MetricsLegacy;
  simulation_logs: SimulationLog[];
  used_ultr: boolean;
  seed?: number;
  ts: number;
  metrics_ce?: Pair;
  metrics_ultr?: Pair;
  lift_ce_vs_base?: { ndcg: number; snips: number };
  lift_ultr_vs_base?: { ndcg: number; snips: number };
  lift_ultr_vs_ce?: { ndcg: number; snips: number };
  ultr_available?: boolean;
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [useUltr, setUseUltr] = useState(false);
  const [seed, setSeed] = useState<string>('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setError(null); setResults(null);
    try {
      const payload: any = { query, use_ultr: useUltr };
      if (seed.trim() !== '') payload.seed = Number(seed);
      const res = await axios.post<SearchResponse>('http://localhost:8000/api/search', payload);
      setResults(res.data);
    } catch (err) {
      setError('Failed to fetch results. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const isCentered = !results && !loading;
  const ultrFlag = results?.used_ultr ?? useUltr;
  const seedShown = results?.seed ?? (seed || '—');

  const showVsCE = ultrFlag && results?.lift_ultr_vs_ce;

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 flex flex-col">
      <div className={`transition-all duration-500 ease-in-out ${isCentered ? 'flex-grow flex flex-col justify-center items-center -mt-20' : 'bg-white shadow-sm py-4 sticky top-0 z-10'}`}>
        <div className={`container mx-auto px-4 ${isCentered ? 'max-w-2xl text-center' : 'max-w-6xl flex items-center gap-6'}`}>
          <h1 className={`font-bold text-blue-600 transition-all duration-300 ${isCentered ? 'text-6xl mb-8' : 'text-2xl flex-shrink-0'}`}>ULTR Search</h1>

          <form onSubmit={handleSearch} className={`relative transition-all duration-300 ${isCentered ? 'w-full' : 'flex-grow max-w-3xl'}`}>
            <div className="relative group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search for anything..."
                className={`w-full px-6 rounded-full border border-gray-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${isCentered ? 'py-4 text-lg shadow-md hover:shadow-lg' : 'py-2.5 text-base bg-gray-50 hover:bg-white hover:shadow-md focus:bg-white'}`}
              />
              <button type="submit" className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 text-blue-600 rounded-full hover:bg-blue-50 transition-colors ${loading ? 'opacity-50' : ''}`} disabled={loading}>
                {loading ? <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div> : <Search size={isCentered ? 24 : 20} />}
              </button>
            </div>
          </form>

          {/* 仅此处改为使用本地 useUltr，点击即刻更新外观与文案 */}
          <button
            type="button"
            onClick={() => setUseUltr(v => !v)}
            className={`ml-4 px-4 py-2 rounded-full border text-sm ${useUltr ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300'} ${isCentered ? 'mt-4' : ''}`}
            title="Toggle ULTR fine-tuned re-ranking"
          >
            ULTR {useUltr ? 'ON' : 'OFF'}
          </button>

          <input
            type="number"
            inputMode="numeric"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="seed (optional)"
            className={`ml-3 w-36 px-3 py-2 rounded-full border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${isCentered ? 'mt-4' : ''}`}
          />
        </div>
      </div>

      {error && (
        <div className="container mx-auto max-w-3xl mt-4 px-4">
          <div className="bg-red-50 text-red-700 border border-red-200 px-4 py-3 rounded-md text-sm">{error}</div>
        </div>
      )}

      {results && (
        <div className="container mx-auto px-4 max-w-6xl py-8 flex flex-col lg:flex-row gap-8 animate-fade-in">
          <div className="lg:w-2/3 space-y-6">
            <div className="flex items-baseline justify-between border-b border-gray-200 pb-2 mb-4">
              <h2 className="text-sm font-medium text-gray-500">Top Results for "<span className="text-gray-800">{query}</span>"</h2>
              <span className="text-xs text-gray-400">{ultrFlag ? 'Ranked by ULTR Cross-Encoder' : 'Ranked by Cross-Encoder'}</span>
            </div>

            {results.results.map((r, i) => (
              <div key={i} className="group bg-white rounded-xl p-5 border border-transparent hover:border-blue-100 hover:shadow-md transition-all duration-200">
                <div className="flex justify-between items-start gap-4 mb-1">
                  <Link to={`/doc/${r.doc_id}`} target="_blank" rel="noopener noreferrer" className="text-xl text-blue-600 hover:underline font-medium leading-snug">
                    Document {r.doc_id}
                  </Link>
                  <span className="shrink-0 text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                    Score: {r.score.toFixed(4)}
                  </span>
                </div>
                <div className="text-green-700 text-xs mb-2 flex items-center gap-1">
                  <Link to={`/doc/${r.doc_id}`} target="_blank" rel="noopener noreferrer" className="hover:underline">
                    www.msmarco-passage.com/doc/{r.doc_id}
                  </Link>
                  <span className="w-0.5 h-0.5 bg-green-700 rounded-full mx-1"></span>
                  <span className="text-gray-400">Cached</span>
                </div>
                <p className="text-gray-600 text-sm leading-relaxed line-clamp-3">{r.text}</p>
              </div>
            ))}
          </div>

          <div className="lg:w-1/3 space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden sticky top-24">
              <div className="bg-gray-50 px-5 py-4 border-b border-gray-200 flex items-center gap-2">
                <BarChart2 size={18} className="text-blue-600" />
                <h3 className="font-semibold text-gray-800">OPE Evaluation</h3>
              </div>

              <div className="p-5 space-y-4">
                {/* 对 baseline 的提升（兼容原逻辑） */}
                <MetricRow
                  title="Oracle nDCG (vs Baseline)"
                  baseline={results.metrics.baseline.ndcg}
                  target={results.metrics.target.ndcg}
                  lift={results.metrics.lift.ndcg}
                  description="True ranking quality vs baseline"
                />
                <MetricRow
                  title="SNIPS (vs Baseline)"
                  baseline={results.metrics.baseline.snips}
                  target={results.metrics.target.snips}
                  lift={results.metrics.lift.snips}
                  description="Estimated value from biased logs"
                />

                {/* ULTR=ON 时，额外显示 vs CE */}
                {showVsCE && (
                  <>
                    <div className="border-t border-gray-100 pt-3"></div>
                    <MetricRow
                      title="Oracle nDCG (ULTR vs CE)"
                      baseline={results.metrics_ce?.ndcg ?? 0}
                      target={results.metrics_ultr?.ndcg ?? 0}
                      lift={results.lift_ultr_vs_ce?.ndcg ?? 0}
                      description="ULTR relative to base CE"
                    />
                    <MetricRow
                      title="SNIPS (ULTR vs CE)"
                      baseline={results.metrics_ce?.snips ?? 0}
                      target={results.metrics_ultr?.snips ?? 0}
                      lift={results.lift_ultr_vs_ce?.snips ?? 0}
                      description="ULTR relative to base CE"
                    />
                  </>
                )}

                <p className="text-xs text-gray-400 mt-2">
                  seed: {seedShown}  •  ULTR: {ultrFlag ? 'ON' : 'OFF'}
                </p>
              </div>

              <div className="border-top border-gray-200"></div>

              <div className="border-t border-gray-200">
                <button onClick={() => setShowLogs(!showLogs)} className="w-full flex items-center justify-between px-5 py-3 text-sm text-gray-600 hover:bg-gray-50 transition-colors">
                  <span className="flex items-center gap-2"><History size={16} /><span>Simulation Logs</span></span>
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
                        <div className="text-gray-600 truncate font-mono bg-white p-1 rounded border border-gray-200">{log.doc_text}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>
      )}

      {isCentered && <div className="text-center pb-8 text-sm text-gray-400">Powered by Unbiased Learning to Rank (ULTR)</div>}
    </div>
  );
};

interface MetricRowProps { title: string; baseline: number; target: number; lift: number; description: string; }
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
        <span className="text-xs text-gray-500 w-16 text-right">{baseline.toFixed(3)}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500" style={{ width: `${Math.min(target * 100, 100)}%` }}></div>
        </div>
        <span className="text-xs text-blue-600 font-bold w-16 text-right">{target.toFixed(3)}</span>
      </div>
      <p className="text-xs text-gray-400 mt-2">{description}</p>
    </div>
  );
};

export default SearchPage;