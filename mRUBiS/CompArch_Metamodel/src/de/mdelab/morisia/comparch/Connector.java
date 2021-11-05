/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Connector</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Connector#isLazyAdaptation <em>Lazy Adaptation</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Connector#getSource <em>Source</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Connector#getTarget <em>Target</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getConnector()
 * @model
 * @generated
 */
public interface Connector extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Lazy Adaptation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Lazy Adaptation</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Lazy Adaptation</em>' attribute.
	 * @see #setLazyAdaptation(boolean)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getConnector_LazyAdaptation()
	 * @model required="true"
	 * @generated
	 */
	boolean isLazyAdaptation();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Connector#isLazyAdaptation <em>Lazy Adaptation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Lazy Adaptation</em>' attribute.
	 * @see #isLazyAdaptation()
	 * @generated
	 */
	void setLazyAdaptation(boolean value);

	/**
	 * Returns the value of the '<em><b>Source</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.RequiredInterface#getConnector <em>Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Source</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Source</em>' container reference.
	 * @see #setSource(RequiredInterface)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getConnector_Source()
	 * @see de.mdelab.morisia.comparch.RequiredInterface#getConnector
	 * @model opposite="connector" required="true" transient="false"
	 * @generated
	 */
	RequiredInterface getSource();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Connector#getSource <em>Source</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Source</em>' container reference.
	 * @see #getSource()
	 * @generated
	 */
	void setSource(RequiredInterface value);

	/**
	 * Returns the value of the '<em><b>Target</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ProvidedInterface#getConnectors <em>Connectors</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Target</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Target</em>' reference.
	 * @see #setTarget(ProvidedInterface)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getConnector_Target()
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getConnectors
	 * @model opposite="connectors" required="true"
	 * @generated
	 */
	ProvidedInterface getTarget();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Connector#getTarget <em>Target</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Target</em>' reference.
	 * @see #getTarget()
	 * @generated
	 */
	void setTarget(ProvidedInterface value);

} // Connector
