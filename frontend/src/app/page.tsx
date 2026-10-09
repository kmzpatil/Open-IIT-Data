"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/stats")
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(console.error);
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query || query.length < 2) return;
    setLoading(true);
    setSearched(true);
    try {
      const res = await fetch(`http://localhost:8000/api/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setResults([]);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans p-6 sm:p-12">
      <header className="max-w-5xl mx-auto mb-16 flex flex-col md:flex-row items-start md:items-baseline justify-between gap-6">
        <div>
          <h1 className="text-xl font-medium tracking-tight text-slate-900">Geocoding_Engine</h1>
          <p className="text-sm text-slate-500 mt-1">Spatial resolution and verification</p>
        </div>
        
        {stats && (
          <div className="flex gap-8">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">Total Addresses</p>
              <p className="font-mono text-lg text-slate-900 tabular-nums">{stats.total_addresses}</p>
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">Tier A Verified</p>
              <p className="font-mono text-lg text-slate-900 tabular-nums">
                {stats.tiers["A_visit_directly"] || 0}
              </p>
            </div>
          </div>
        )}
      </header>

      <main className="max-w-5xl mx-auto space-y-12">
        <section>
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
            <input 
              type="text" 
              placeholder="Search by Address ID (e.g. AD000001) or location text..." 
              className="flex-1 bg-white border border-slate-200 text-slate-900 rounded-md px-4 py-2.5 text-sm focus:outline-none focus:border-slate-400 focus:ring-1 focus:ring-slate-400 transition-shadow placeholder:text-slate-400 shadow-sm"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button 
              type="submit" 
              className="bg-slate-900 text-white hover:bg-slate-800 font-medium text-sm py-2.5 px-6 rounded-md transition-colors shadow-sm disabled:opacity-50"
              disabled={loading}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </form>
        </section>

        {searched && (
          <section className="animate-in fade-in duration-300">
            <div className="mb-4 flex items-baseline justify-between">
              <h2 className="text-sm font-medium text-slate-900">
                Results
              </h2>
              <span className="font-mono text-[11px] text-slate-500">{results.length} found</span>
            </div>
            
            <div className="overflow-x-auto">
              {results.length > 0 ? (
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-y border-slate-200">
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500">ID</th>
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Address</th>
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Tier</th>
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500 text-right">Coordinates</th>
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500 text-right">Radius (m)</th>
                      <th className="py-3 px-4 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Directions</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm">
                    {results.map((r: any, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                        <td className="py-3 px-4 font-mono text-slate-600">{r.address_id}</td>
                        <td className="py-3 px-4 text-slate-800 max-w-[300px] truncate" title={r.address_text}>
                          {r.address_text}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <span className={`h-1.5 w-1.5 rounded-full ${
                              String(r.tier).startsWith('A') ? 'bg-emerald-500' : 
                              String(r.tier).startsWith('B') ? 'bg-amber-500' : 'bg-red-500'
                            }`} />
                            <span className="text-slate-800">{r.tier}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-600 tabular-nums text-right whitespace-nowrap">
                          {Number(r.pin_x).toFixed(1)}, {Number(r.pin_y).toFixed(1)}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-600 tabular-nums text-right">
                          {Number(r.radius_90_m).toFixed(1)}
                        </td>
                        <td className="py-3 px-4 text-slate-500 max-w-[200px] truncate" title={r.directions}>
                          {r.directions || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="py-12 text-center border border-slate-200 rounded-md bg-slate-50">
                  <p className="text-sm text-slate-500">No addresses found for "{query}".</p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

