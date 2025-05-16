"use client";
import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, Loader2, RefreshCw, Mic, MicOff, FileText, Download } from "lucide-react";
import { format } from 'date-fns';
import { Interweave } from 'interweave';
// Add these imports for matchers
import { UrlMatcher, HashtagMatcher } from 'interweave-autolink';

interface Message {
  type: 'user' | 'bot';
  content: string;
  is_fir?: boolean;
  pdf_url?: string;
  fir_number?: string;
  timestamp: string; // Add timestamp
}

export default function Home() {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'bot',
      content: `👋 Hi! I'm your CopBot.
      
I can help you with Police Act, Standing Orders, IPC, and case data. How can I help?`,
      timestamp: new Date().toISOString(), // Set initial timestamp
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingState, setLoadingState] = useState<string>('');
  const [isListening, setIsListening] = useState(false);
  const loadingStates = [
    "🔍 Searching relevant documents...",
    "📊 Analyzing information...",
    "⚖️ Cross-referencing legal data...",
    "📝 Preparing response..."
  ];

  useEffect(() => {
    let currentIndex = 0;
    let interval: NodeJS.Timeout;

    if (isLoading) {
      interval = setInterval(() => {
        setLoadingState(loadingStates[currentIndex]);
        currentIndex = (currentIndex + 1) % loadingStates.length;
      }, 2000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isLoading]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { type: 'user', content: input, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setLoadingState(loadingStates[0]);

    try {
      const response = await fetch(`http://localhost:5000/query?query=${encodeURIComponent(input)}`);
      const data = await response.json();

      setMessages(prev => [...prev, {
        type: 'bot',
        content: data.response,
        is_fir: data.is_fir,
        pdf_url: data.pdf_url,
        fir_number: data.fir_number,
        timestamp: new Date().toISOString(), // Set timestamp for bot reply
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Server is down',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
      setLoadingState('');
    }
  };

  const handleReset = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/reset', {
        method: 'POST'
      });
      if (response.ok) {
        setMessages([
          {
            type: 'bot',
            content: `👋 Hi! I'm your CopBot.
      
I can help you with Police Act, Standing Orders, IPC, and case data. How can I help?`,
            timestamp: new Date().toISOString(),
          }
        ]);
      }
    } catch (error) {
      console.error('Error resetting chat:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }

    if (!recognitionRef.current) {
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => setIsListening(true);
      recognition.onend = () => setIsListening(false);
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
      recognition.onresult = (event: any) => {
        const transcript: string = event.results[0][0].transcript;
        setInput(transcript);
      };

      recognitionRef.current = recognition;
    }

    if (!isListening) {
      recognitionRef.current.start();
    } else {
      recognitionRef.current.stop();
    }
  };

  const handlePdfDownload = (url: string) => {
    const fullUrl = `http://localhost:5000${url}`;
    const link = document.createElement('a');
    link.href = fullUrl;
    link.target = '_blank';
    link.download = 'FIR.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-[conic-gradient(at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-100 to-gray-900">
      <nav className="w-full bg-white/95 shadow-lg backdrop-blur-sm p-4 sticky top-0 z-50 border-b border-black/10">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold text-slate-800">CopBot</h1>
          <Link
            href="/dashboard"
            className="text-slate-600 hover:text-slate-900 transition-colors"
          >
            Go to Dashboard
          </Link>
        </div>
      </nav>

      <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
        <Card className="w-full max-w-4xl border-2 border-black/10 bg-white/90 backdrop-blur-lg shadow-[0_8px_32px_rgb(0,0,0,0.15)] hover:shadow-[0_8px_32px_rgb(0,0,0,0.20)] transition-shadow">
          <CardHeader className="border-b border-black/10 backdrop-blur-lg bg-white/95 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 relative rounded-full overflow-hidden ring-2 ring-slate-200">
                  <Image
                    src="/police-badge.jpg"
                    alt="Police Badge"
                    fill
                    className="object-cover transform hover:scale-110 transition-transform duration-200"
                  />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                    CopBot
                  </h1>
                  <p className="text-slate-600 flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    Online
                  </p>
                </div>
              </div>
              <Button
                onClick={handleReset}
                variant="outline"
                size="icon"
                className="text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
                disabled={isLoading}
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="p-0">
            <ScrollArea className="h-[500px] p-6 shadow-inner border-black/5" ref={scrollAreaRef}>
              <div className="space-y-6">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex items-start gap-3 ${
                      message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
                    } animate-in slide-in-from-bottom-5 hover:translate-y-[-2px] transition-all duration-200`}
                  >
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      message.type === 'user' ? 'bg-gradient-to-br from-slate-700 to-slate-900' : 'bg-gradient-to-br from-slate-100 to-slate-200'
                    } shadow-lg`}>
                      {message.type === 'user' ? 
                        <User className="w-6 h-6 text-white" /> : 
                        <Bot className="w-6 h-6 text-slate-700" />
                      }
                    </div>
                    <div className="flex flex-col gap-1 max-w-[80%]">
                      <div
                        className={`rounded-2xl p-4 whitespace-pre-wrap relative ${
                          message.type === 'user'
                            ? 'bg-gradient-to-br from-slate-700 to-slate-800 text-white border border-black/20'
                            : 'bg-gradient-to-br from-slate-50 to-slate-100 text-slate-800 border border-black/10'
                        } shadow-lg hover:shadow-xl transition-shadow`}
                      >
                        <Interweave
                          content={message.content}
                          matchers={[new UrlMatcher('url'), new HashtagMatcher('hashtag')]}
                        />
                        {message.is_fir && message.pdf_url && (
                          <div className="absolute top-2 right-2 flex gap-2">
                            <Button
                              onClick={() => handlePdfDownload(message.pdf_url)}
                              className="bg-red-500 hover:bg-red-600 p-2 rounded-full"
                              size="icon"
                              variant="ghost"
                              title="Download PDF"
                            >
                              <Download className="w-4 h-4 text-white" />
                            </Button>
                          </div>
                        )}
                      </div>
                      <span className={`text-xs ${
                        message.type === 'user' ? 'text-right' : ''
                      } text-slate-500`}>
                        {format(new Date(message.timestamp), 'HH:mm')}
                      </span>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex items-start gap-3 animate-in fade-in-50">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center shadow-md">
                      <Bot className="w-6 h-6 text-slate-700" />
                    </div>
                    <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-2xl p-4 flex items-center gap-3 shadow-md">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.2s]"></span>
                        <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></span>
                      </div>
                      <span className="text-slate-600">
                        {loadingState}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>

          <CardFooter className="border-t border-black/10 backdrop-blur-lg bg-white/95 p-4 shadow-md">
            <div className="flex w-full gap-3">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Type your message here..."
                className="bg-white/80 border-black/20 focus:border-black/30 focus:ring-black/20 text-slate-900 placeholder:text-slate-400 shadow-sm hover:shadow transition-shadow"
              />
              <Button
                onClick={handleVoiceInput}
                className={`gap-2 ${isListening ? 'bg-red-500' : 'bg-gradient-to-r from-slate-700 to-slate-900'} hover:from-slate-800 hover:to-slate-900 transition-all duration-200 shadow-md`}
              >
                {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                {isListening ? 'Listening...' : 'Voice'}
              </Button>
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="gap-2 bg-gradient-to-r from-slate-700 to-slate-900 hover:from-slate-800 hover:to-slate-900 transition-all duration-200 shadow-md"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Send
              </Button>
            </div>
          </CardFooter>
        </Card>
      </main>
    </div>
  );
}
