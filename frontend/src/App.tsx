import React from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { ClassifierWorkspace } from './components/ClassifierWorkspace';
import { ModelInsights } from './components/ModelInsights';
import { ClassesGrid } from './components/ClassesGrid';
import { Footer } from './components/Footer';

export const App: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-space text-slate-100">
      <Navbar />
      <main>
        <Hero />
        <ClassifierWorkspace />
        <ModelInsights />
        <ClassesGrid />
      </main>
      <Footer />
    </div>
  );
};

export default App;
