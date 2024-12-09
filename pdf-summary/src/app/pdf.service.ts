import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class PdfService {

  private apiUrl = 'http://localhost:4050/summarize';

  constructor(private http: HttpClient) { }

  uploadFile(file: File, wordLimit: number) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('word_limit', wordLimit.toString());

    return this.http.post<{ summary: string }>(this.apiUrl, formData);
  }
}
