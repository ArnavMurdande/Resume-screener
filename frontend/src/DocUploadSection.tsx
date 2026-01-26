import React, { useRef } from 'react';
import { Upload, FileText, User, Loader2, FileCheck, AlertCircle, Trash2 } from 'lucide-react';

interface DocUploadProps {
  title: string;
  type: 'resume' | 'jd';
  file: File | null;
  setFile: (file: File | null) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
  status: 'idle' | 'uploading' | 'success' | 'error';
  setStatus: (status: 'idle' | 'uploading' | 'success' | 'error') => void;
  onUpload: (file: File, type: 'resume' | 'jd') => Promise<void> | void;
}

const DocUploadSection: React.FC<DocUploadProps> = ({ 
  title, 
  type, 
  file, 
  setFile, 
  setPreviewUrl, 
  previewUrl,
  status, 
  setStatus,
  onUpload 
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url); 
      onUpload(selectedFile, type);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
       // CHANGED: Allow PDF OR Text/Plain
       if (selectedFile.type === "application/pdf" || selectedFile.type === "text/plain" || selectedFile.name.endsWith(".txt")) {
          setFile(selectedFile);
          // Create object URL for preview (iframe works for txt too usually, or show icon)
          const url = URL.createObjectURL(selectedFile);
          setPreviewUrl(url);
          onUpload(selectedFile, type);
       } else {
         alert("Please upload a PDF or TXT file.");
       }
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreviewUrl(null);
    setStatus('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="doc-section glass-panel">
      <div className="doc-header">
        <h2 className="flex-center gap-2">
          {type === 'resume' ? <User className="text-blue-400" size={18} /> : <FileText className="text-secondary" size={18} />}
          {title}
        </h2>
        <div className="flex-center gap-2">
            {status === 'uploading' && <span className="status-label text-blue-400 animate-pulse">Uploading...</span>}
            {status === 'success' && <span className="status-label text-success flex-center gap-1"><FileCheck size={12}/> Ready</span>}
            {status === 'error' && <span className="status-label text-error flex-center gap-1"><AlertCircle size={12}/> Failed</span>}
            {file && <button onClick={clearFile} className="icon-btn-danger"><Trash2 size={16}/></button>}
        </div>
      </div>

      <div className="doc-body">
        {!file || !previewUrl ? (
          <div 
            className="drop-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="drop-icon-wrapper">
               <Upload size={32} />
            </div>
            <p className="drop-title">Drag & Drop PDF</p>
            <p className="drop-subtitle">or click to browse</p>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileSelect} 
              accept=".pdf,.txt"   // CHANGED: Added .txt
              className="hidden" 
            />
          </div>
        ) : (
          <div className="preview-container">
             <iframe 
                src={previewUrl} 
                className="pdf-preview" 
                title={`${title} Preview`}
             />
          </div>
        )}
        
        {status === 'uploading' && (
             <div className="loading-overlay">
                <Loader2 className="animate-spin text-primary mb-2" size={40} />
                <p className="text-primary font-medium">Processing Document...</p>
             </div>
        )}
      </div>
    </div>
  );
};

export default DocUploadSection;
