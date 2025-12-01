import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchPage from './SearchPage';
import DocumentPage from './DocumentPage';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/doc/:id" element={<DocumentPage />} />
      </Routes>
    </Router>
  );
};

export default App;
