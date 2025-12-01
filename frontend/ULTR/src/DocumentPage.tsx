import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { FileText, AlertCircle } from 'lucide-react';

interface Document {
  doc_id: string;
  text: string;
}

const DocumentPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/document/${id}`);
        setDocument(response.data);
      } catch (err) {
        setError('Document not found or failed to load.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchDocument();
    }
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full text-center">
          <div className="flex justify-center mb-4 text-red-500">
            <AlertCircle size={48} />
          </div>
          <h2 className="text-xl font-bold text-gray-800 mb-2">Error Loading Document</h2>
          <p className="text-gray-600">{error || 'Document not found.'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto bg-white shadow-sm rounded-lg overflow-hidden border border-gray-200">
        {/* Header */}
        <div className="bg-blue-50 px-6 py-4 border-b border-blue-100 flex items-center gap-3">
          <FileText className="text-blue-600" size={24} />
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Document Viewer</h1>
            <p className="text-sm text-blue-600 font-mono">{document.doc_id}</p>
          </div>
        </div>

        {/* Content */}
        <div className="p-8">
          <div className="prose max-w-none">
            <p className="text-gray-800 leading-relaxed text-lg">
              {document.text}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t border-gray-100 text-sm text-gray-500 flex justify-between items-center">
          <span>Source: MS MARCO Passage Corpus</span>
          <span className="font-mono text-xs bg-gray-200 px-2 py-1 rounded">ID: {document.doc_id}</span>
        </div>
      </div>
    </div>
  );
};

export default DocumentPage;
