"use client";
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

const Dashboard = () => {
    const [files, setFiles] = useState<Record<string, { description: string; path: string }>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [uploading, setUploading] = useState<string | null>(null);
    const [fileToReplace, setFileToReplace] = useState<{ key: string, file: File } | null>(null);
    const [editingDescription, setEditingDescription] = useState<{ key: string; description: string } | null>(null);
    const [rebuildConfirmOpen, setRebuildConfirmOpen] = useState(false);
    const [isRebuilding, setIsRebuilding] = useState(false);

    useEffect(() => {
        const loadFiles = async () => {
            try {
                const res = await fetch("http://127.0.0.1:5000/get_files");
                const data = await res.json();
                console.log("Received data:", data);
                setFiles(data);
            } catch (err) {
                console.error("Error fetching files:", err);
                setError("Failed to load files.");
            } finally {
                setLoading(false);
            }
        };
        loadFiles();
    }, []);

    const handleFileUpload = async (fileKey: string, file: File) => {
        setFileToReplace({ key: fileKey, file });
    };

    const confirmReplace = async () => {
        if (!fileToReplace) return;

        setUploading(fileToReplace.key);
        const formData = new FormData();
        formData.append('file', fileToReplace.file);

        try {
            const response = await fetch(`http://127.0.0.1:5000/replace/${fileToReplace.key}`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const res = await fetch("http://127.0.0.1:5000/get_files");
            const data = await res.json();
            setFiles(data);
        } catch (err) {
            console.error("Error uploading file:", err);
            setError("Failed to replace file.");
        } finally {
            setUploading(null);
            setFileToReplace(null);
        }
    };

    const handleEditDescription = async () => {
        if (!editingDescription) return;

        try {
            const response = await fetch(`http://127.0.0.1:5000/edit_description/${editingDescription.key}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ description: editingDescription.description }),
            });

            if (!response.ok) throw new Error('Failed to update description');

            const res = await fetch("http://127.0.0.1:5000/get_files");
            const data = await res.json();
            setFiles(data);
            toast.success("Description updated successfully");
        } catch (err) {
            console.error("Error updating description:", err);
            toast.error("Failed to update description");
        } finally {
            setEditingDescription(null);
        }
    };

    const handleRebuildIndexes = async () => {
        setIsRebuilding(true);
        try {
            const response = await fetch('http://127.0.0.1:5000/rebuild_indexes', {
                method: 'POST',
            });

            if (!response.ok) throw new Error('Failed to rebuild indexes');

            toast.success("Indexes rebuilt successfully");
        } catch (err) {
            console.error("Error rebuilding indexes:", err);
            toast.error("Failed to rebuild indexes");
        } finally {
            setIsRebuilding(false);
            setRebuildConfirmOpen(false);
        }
    };

    const filteredFiles = Object.entries(files).filter(([key, value]) =>
        key.toLowerCase().includes(searchQuery.toLowerCase()) ||
        value.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading) {
        return (
            <div className="flex h-screen items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-screen items-center justify-center">
                <p className="text-red-500">Error: {error}</p>
            </div>
        );
    }

    return (
        <>
            <div className="min-h-screen bg-[conic-gradient(at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-100 to-gray-900">
                <nav className="w-full bg-white/95 shadow-lg backdrop-blur-sm p-4 sticky top-0 z-50 border-b border-black/10">
                    <div className="max-w-4xl mx-auto flex justify-between items-center">
                        <h1 className="text-xl font-bold text-slate-800">Legal Documents</h1>
                        <div className="flex gap-4">
                            <Button
                                variant="outline"
                                onClick={() => setRebuildConfirmOpen(true)}
                                disabled={isRebuilding}
                            >
                                {isRebuilding ? (
                                    <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Rebuilding...</>
                                ) : (
                                    'Rebuild Indexes'
                                )}
                            </Button>
                            <Link
                                href="/"
                                className="text-slate-600 hover:text-slate-900 transition-colors"
                            >
                                Back to Chat
                            </Link>
                        </div>
                    </div>
                </nav>

                <div className="container mx-auto p-6">
                    <div className="mb-8 space-y-4">
                        <h1 className="text-4xl font-bold text-center bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                            📚 Legal Document Dashboard
                        </h1>
                        <div className="max-w-md mx-auto">
                            <Input
                                type="search"
                                placeholder="Search documents..."
                                className="bg-white/80 border-black/20 focus:border-black/30 focus:ring-black/20 text-slate-900 placeholder:text-slate-400"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {filteredFiles.map(([key, value]) => (
                            <Card 
                                key={key} 
                                className="group hover:scale-105 transition-all duration-200 bg-white border-none shadow-lg hover:shadow-xl"
                            >
                                <CardHeader>
                                    <h3 className="text-xl font-semibold text-gray-800 group-hover:text-blue-600">
                                        {key}
                                    </h3>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-600">{value.description}</p>
                                    <div className="mt-4 text-sm text-gray-500">
                                        📂 {value.path}
                                    </div>
                                    <div className="mt-4 space-y-2">
                                        <Button
                                            variant="outline"
                                            className="w-full"
                                            onClick={() => setEditingDescription({ key, description: value.description })}
                                        >
                                            Edit Description
                                        </Button>
                                        <Input
                                            type="file"
                                            accept=".pdf"
                                            onChange={(e) => {
                                                const file = e.target.files?.[0];
                                                if (file) {
                                                    handleFileUpload(key, file);
                                                }
                                            }}
                                            disabled={uploading === key}
                                            className="hidden"
                                            id={`file-${key}`}
                                        />
                                        <Button
                                            onClick={() => document.getElementById(`file-${key}`)?.click()}
                                            disabled={uploading === key}
                                            className="w-full"
                                        >
                                            {uploading === key ? (
                                                <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Uploading...</>
                                            ) : (
                                                'Replace File'
                                            )}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </div>

            <Dialog open={!!editingDescription} onOpenChange={() => setEditingDescription(null)}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Edit Description</DialogTitle>
                    </DialogHeader>
                    <Textarea
                        value={editingDescription?.description || ''}
                        onChange={(e) => setEditingDescription(prev => 
                            prev ? { ...prev, description: e.target.value } : null
                        )}
                        className="min-h-[100px]"
                    />
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setEditingDescription(null)}>
                            Cancel
                        </Button>
                        <Button onClick={handleEditDescription}>
                            Save Changes
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <AlertDialog open={rebuildConfirmOpen} onOpenChange={setRebuildConfirmOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Rebuild All Indexes</AlertDialogTitle>
                        <AlertDialogDescription>
                            This will rebuild all document indexes. This process may take some time and the chatbot may be temporarily unavailable. Are you sure you want to continue?
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={handleRebuildIndexes} disabled={isRebuilding}>
                            {isRebuilding ? (
                                <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Rebuilding...</>
                            ) : (
                                'Continue'
                            )}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            <AlertDialog open={!!fileToReplace} onOpenChange={() => setFileToReplace(null)}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Confirm File Replacement</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to replace this file? This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={confirmReplace}>Continue</AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </>
    );
};

export default Dashboard;