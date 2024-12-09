import { Component } from '@angular/core';
import { PdfService } from './pdf.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'pdf-summary';
  selectedFile: File | null = null;
  wordLimit: number = 100;
  result: string = '';

  constructor(private pdfService: PdfService) { }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  uploadFile(event: any) {
    event.preventDefault();
    if (this.selectedFile) {
      this.pdfService.uploadFile(this.selectedFile, this.wordLimit).subscribe(
        (response) => {
          console.log('Summary:', response.summary);
          this.result = response.summary;
          // alert(`Summary: ${response.summary}`);
        },
        (error) => {
          console.error('Error:', error);
          alert('Failed to summarize the PDF.');
        }
      );
    } else {
      alert('Please select a file first.');
    }
  }
}
