"""
Security Testing with Bandit for ZeroGravity

This module implements security tests using Bandit to identify
vulnerabilities in the ZeroGravity platform codebase.
"""
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import xml.etree.ElementTree as ET


class BanditScanner:
    """Bandit security scanner for ZeroGravity codebase"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Scan a directory for security vulnerabilities using Bandit
        
        Args:
            directory: Path to directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            Dictionary containing scan results
        """
        try:
            # Build the bandit command
            cmd = ["bandit", "-r" if recursive else "-f", "json", directory]
            
            if self.config_file:
                cmd.extend(["-c", self.config_file])
            
            # Execute the scan
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # No issues found, but Bandit still returns results
                scan_results = json.loads(result.stdout)
            elif result.returncode == 1:
                # Issues found, which is normal
                scan_results = json.loads(result.stdout)
            else:
                # Actual error occurred
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
            
            return scan_results
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Bandit scan timed out for directory: {directory}")
            return {
                "errors": ["Scan timed out"],
                "results": []
            }
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Bandit scan failed: {e}")
            return {
                "errors": [str(e)],
                "results": []
            }
        except json.JSONDecodeError:
            self.logger.error("Failed to parse Bandit output")
            return {
                "errors": ["Failed to parse Bandit output"],
                "results": []
            }
        except FileNotFoundError:
            self.logger.error("Bandit is not installed or not in PATH")
            return {
                "errors": ["Bandit is not installed or not in PATH"],
                "results": []
            }
    
    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """
        Scan a single file for security vulnerabilities
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            Dictionary containing scan results
        """
        return self.scan_directory(file_path, recursive=False)
    
    def generate_report(self, scan_results: Dict[str, Any], output_format: str = "text") -> str:
        """
        Generate a security report from scan results
        
        Args:
            scan_results: Results from bandit scan
            output_format: Format for report ('text', 'json', 'html', 'xml')
            
        Returns:
            Formatted security report
        """
        if output_format == "text":
            return self._format_text_report(scan_results)
        elif output_format == "json":
            return json.dumps(scan_results, indent=2)
        elif output_format == "html":
            return self._format_html_report(scan_results)
        elif output_format == "xml":
            return self._format_xml_report(scan_results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _format_text_report(self, scan_results: Dict[str, Any]) -> str:
        """Format scan results as text report"""
        report_lines = []
        report_lines.append("ZeroGravity Security Scan Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Scan Time: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Add summary
        metrics = scan_results.get("metrics", {})
        if metrics:
            total_files = metrics.get("_totals", {}).get("loc", 0)
            total_issues = sum([
                metrics.get("_totals", {}).get("SEVERITY.HIGH", 0),
                metrics.get("_totals", {}).get("SEVERITY.MEDIUM", 0),
                metrics.get("_totals", {}).get("SEVERITY.LOW", 0)
            ])
            
            report_lines.append(f"Files scanned: {total_files}")
            report_lines.append(f"Total issues found: {total_issues}")
            report_lines.append(f"High severity: {metrics.get('_totals', {}).get('SEVERITY.HIGH', 0)}")
            report_lines.append(f"Medium severity: {metrics.get('_totals', {}).get('SEVERITY.MEDIUM', 0)}")
            report_lines.append(f"Low severity: {metrics.get('_totals', {}).get('SEVERITY.LOW', 0)}")
            report_lines.append("")
        
        # Add detailed results
        results = scan_results.get("results", [])
        if results:
            report_lines.append("Detailed Findings:")
            report_lines.append("-" * 20)
            
            for result in results:
                report_lines.append(f"File: {result.get('filename', 'Unknown')}")
                report_lines.append(f"Line: {result.get('line_number', 'Unknown')}")
                report_lines.append(f"Issue: {result.get('issue_text', 'Unknown')}")
                report_lines.append(f"Severity: {result.get('issue_severity', 'Unknown')}")
                report_lines.append(f"Confidence: {result.get('issue_confidence', 'Unknown')}")
                report_lines.append(f"Code: {result.get('code', 'N/A')[:100]}...")  # Truncate long code
                report_lines.append("")
        
        if scan_results.get("errors"):
            report_lines.append("Errors during scan:")
            report_lines.append("-" * 20)
            for error in scan_results["errors"]:
                report_lines.append(f"- {error}")
        
        return "\n".join(report_lines)
    
    def _format_html_report(self, scan_results: Dict[str, Any]) -> str:
        """Format scan results as HTML report"""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head><title>ZeroGravity Security Scan Report</title></head>",
            "<body>",
            "<h1>ZeroGravity Security Scan Report</h1>",
            f"<p>Scan Time: {datetime.now().isoformat()}</p>"
        ]
        
        # Add summary
        metrics = scan_results.get("metrics", {})
        if metrics:
            html.append("<h2>Summary</h2>")
            html.append("<ul>")
            html.append(f"<li>Files scanned: {metrics.get('_totals', {}).get('loc', 0)}</li>")
            total_issues = sum([
                metrics.get("_totals", {}).get("SEVERITY.HIGH", 0),
                metrics.get("_totals", {}).get("SEVERITY.MEDIUM", 0),
                metrics.get("_totals", {}).get("SEVERITY.LOW", 0)
            ])
            html.append(f"<li>Total issues found: {total_issues}</li>")
            html.append(f"<li>High severity: {metrics.get('_totals', {}).get('SEVERITY.HIGH', 0)}</li>")
            html.append(f"<li>Medium severity: {metrics.get('_totals', {}).get('SEVERITY.MEDIUM', 0)}</li>")
            html.append(f"<li>Low severity: {metrics.get('_totals', {}).get('SEVERITY.LOW', 0)}</li>")
            html.append("</ul>")
        
        # Add detailed results
        results = scan_results.get("results", [])
        if results:
            html.append("<h2>Detailed Findings</h2>")
            for result in results:
                html.append("<div style='border: 1px solid #ccc; margin: 10px; padding: 10px;'>")
                html.append(f"<h3>File: {result.get('filename', 'Unknown')}</h3>")
                html.append(f"<p><strong>Line:</strong> {result.get('line_number', 'Unknown')}</p>")
                html.append(f"<p><strong>Issue:</strong> {result.get('issue_text', 'Unknown')}</p>")
                html.append(f"<p><strong>Severity:</strong> {result.get('issue_severity', 'Unknown')}</p>")
                html.append(f"<p><strong>Confidence:</strong> {result.get('issue_confidence', 'Unknown')}</p>")
                html.append(f"<pre>{result.get('code', 'N/A')}</pre>")
                html.append("</div>")
        
        if scan_results.get("errors"):
            html.append("<h2>Errors</h2>")
            html.append("<ul>")
            for error in scan_results["errors"]:
                html.append(f"<li>{error}</li>")
            html.append("</ul>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _format_xml_report(self, scan_results: Dict[str, Any]) -> str:
        """Format scan results as XML report"""
        root = ET.Element("security_report")
        root.set("timestamp", datetime.now().isoformat())
        root.set("tool", "bandit")
        
        # Add summary
        summary = ET.SubElement(root, "summary")
        metrics = scan_results.get("metrics", {})
        if metrics:
            totals = metrics.get("_totals", {})
            summary.set("files_scanned", str(totals.get("loc", 0)))
            total_issues = sum([
                totals.get("SEVERITY.HIGH", 0),
                totals.get("SEVERITY.MEDIUM", 0),
                totals.get("SEVERITY.LOW", 0)
            ])
            summary.set("total_issues", str(total_issues))
            summary.set("high_severity", str(totals.get("SEVERITY.HIGH", 0)))
            summary.set("medium_severity", str(totals.get("SEVERITY.MEDIUM", 0)))
            summary.set("low_severity", str(totals.get("SEVERITY.LOW", 0)))
        
        # Add detailed results
        results_elem = ET.SubElement(root, "results")
        for result in scan_results.get("results", []):
            issue = ET.SubElement(results_elem, "issue")
            issue.set("filename", result.get("filename", "Unknown"))
            issue.set("line_number", str(result.get("line_number", "Unknown")))
            issue.set("severity", result.get("issue_severity", "Unknown"))
            issue.set("confidence", result.get("issue_confidence", "Unknown"))
            
            issue_text = ET.SubElement(issue, "issue_text")
            issue_text.text = result.get("issue_text", "Unknown")
            
            code = ET.SubElement(issue, "code")
            code.text = result.get("code", "N/A")
        
        # Add errors
        if scan_results.get("errors"):
            errors_elem = ET.SubElement(root, "errors")
            for error in scan_results["errors"]:
                error_elem = ET.SubElement(errors_elem, "error")
                error_elem.text = error
        
        # Convert to string
        return ET.tostring(root, encoding="unicode")


class SecurityTester:
    """Main security testing class for ZeroGravity"""
    
    def __init__(self, bandit_config: Optional[str] = None):
        self.bandit_scanner = BanditScanner(bandit_config)
        self.logger = logging.getLogger(__name__)
    
    def run_security_scan(self, target_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive security scan on ZeroGravity codebase
        
        Args:
            target_path: Path to scan (defaults to entire zero_gravity_core)
            
        Returns:
            Dictionary containing scan results and summary
        """
        if target_path is None:
            # Default to scanning the entire zero_gravity_core directory
            target_path = str(Path(__file__).parent.parent)
        
        self.logger.info(f"Starting security scan of: {target_path}")
        
        # Perform the scan
        scan_results = self.bandit_scanner.scan_directory(target_path)
        
        # Generate summary
        summary = self._generate_security_summary(scan_results)
        
        # Create comprehensive report
        report = {
            "scan_time": datetime.now().isoformat(),
            "target_path": target_path,
            "summary": summary,
            "raw_results": scan_results,
            "report_text": self.bandit_scanner.generate_report(scan_results, "text"),
            "report_html": self.bandit_scanner.generate_report(scan_results, "html")
        }
        
        self.logger.info(f"Security scan completed. Found {summary['total_issues']} issues.")
        
        return report
    
    def _generate_security_summary(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of security scan results"""
        metrics = scan_results.get("metrics", {}).get("_totals", {})
        
        total_issues = sum([
            metrics.get("SEVERITY.HIGH", 0),
            metrics.get("SEVERITY.MEDIUM", 0),
            metrics.get("SEVERITY.LOW", 0)
        ])
        
        return {
            "total_files_scanned": metrics.get("loc", 0),
            "total_issues": total_issues,
            "high_severity": metrics.get("SEVERITY.HIGH", 0),
            "medium_severity": metrics.get("SEVERITY.MEDIUM", 0),
            "low_severity": metrics.get("SEVERITY.LOW", 0),
            "errors_during_scan": len(scan_results.get("errors", []))
        }
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "security_reports") -> Dict[str, str]:
        """
        Save security scan report to files
        
        Args:
            report: Security scan report from run_security_scan
            output_dir: Directory to save reports to
            
        Returns:
            Dictionary with paths to saved report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save different formats
        report_files = {}
        
        # Text report
        text_path = output_path / f"security_report_{timestamp}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report["report_text"])
        report_files["text"] = str(text_path)
        
        # JSON report
        json_path = output_path / f"security_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report["raw_results"], f, indent=2)
        report_files["json"] = str(json_path)
        
        # HTML report
        html_path = output_path / f"security_report_{timestamp}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(report["report_html"])
        report_files["html"] = str(html_path)
        
        # XML report
        xml_path = output_path / f"security_report_{timestamp}.xml"
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(self.bandit_scanner.generate_report(report["raw_results"], "xml"))
        report_files["xml"] = str(xml_path)
        
        self.logger.info(f"Security reports saved to: {output_dir}")
        
        return report_files
    
    def check_for_critical_issues(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check scan results for critical security issues
        
        Args:
            report: Security scan report from run_security_scan
            
        Returns:
            List of critical issues found
        """
        critical_issues = []
        
        for result in report["raw_results"].get("results", []):
            if result.get("issue_severity") == "HIGH":
                # Additional checks for specific critical vulnerabilities
                issue_text = result.get("issue_text", "").lower()
                
                # Check for specific critical vulnerabilities
                critical_keywords = [
                    "injection", "xss", "command injection", "path traversal", 
                    "deserialization", "hardcoded", "credentials", "password"
                ]
                
                is_critical = any(keyword in issue_text for keyword in critical_keywords)
                
                if is_critical:
                    critical_issues.append(result)
        
        return critical_issues


def run_security_audit() -> Dict[str, Any]:
    """
    Run a comprehensive security audit of the ZeroGravity platform
    
    Returns:
        Dictionary containing audit results
    """
    security_tester = SecurityTester()
    return security_tester.run_security_scan()


def save_security_report(report: Dict[str, Any], output_dir: str = "security_reports") -> Dict[str, str]:
    """
    Save a security report to disk
    
    Args:
        report: Security report from run_security_audit
        output_dir: Directory to save reports to
        
    Returns:
        Dictionary with paths to saved report files
    """
    security_tester = SecurityTester()
    return security_tester.save_report(report, output_dir)


# Example usage and testing
if __name__ == "__main__":
    # Run a quick security check
    print("Running ZeroGravity security audit...")
    
    try:
        report = run_security_audit()
        
        print(f"Scan completed at: {report['scan_time']}")
        print(f"Total issues found: {report['summary']['total_issues']}")
        print(f"High severity: {report['summary']['high_severity']}")
        print(f"Medium severity: {report['summary']['medium_severity']}")
        print(f"Low severity: {report['summary']['low_severity']}")
        
        # Check for critical issues
        critical_issues = []
        for result in report["raw_results"].get("results", []):
            if result.get("issue_severity") == "HIGH":
                critical_issues.append(result)
        
        print(f"Critical issues: {len(critical_issues)}")
        
        if critical_issues:
            print("\nTop critical issues:")
            for i, issue in enumerate(critical_issues[:5]):  # Show top 5
                print(f"  {i+1}. {issue.get('filename')}:{issue.get('line_number')}")
                print(f"     {issue.get('issue_text')[:100]}...")
        
        # Save the report
        report_files = save_security_report(report)
        print(f"\nReports saved to: {list(report_files.values())}")
        
    except Exception as e:
        print(f"Security audit failed: {e}")
        print("Make sure Bandit is installed: pip install bandit")
